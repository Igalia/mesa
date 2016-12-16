/*
 * Copyright © 2016 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <intel_bufmgr.h>

#include "brw_context.h"
#include "intel_screen.h"
#include "intel_batchbuffer.h"
#include "brw_defines.h"

#define __BEGIN_BATCH(n) do {                        \
   uint32_t *___map = ctx->batch->map_next;          \
   ctx->batch->map_next += (n);

#define __OUT_BATCH(d) *___map++ = (d)

#define __ADVANCE_BATCH()   \
} while (0)

#define __OUT_RELOC(buf, delta) do { \
   uint32_t __offset = (___map - ctx->batch->map) * 4;            \
   __OUT_BATCH(batchbuffer_reloc(ctx->batch, (buf), __offset,     \
                                 (I915_GEM_DOMAIN_INSTRUCTION),   \
                                 (I915_GEM_DOMAIN_INSTRUCTION),   \
                                 (delta)));                       \
} while (0)

struct check_register_writes_context {
   dri_bufmgr *bufmgr;
   drm_intel_context *hw_ctx;
   struct intel_batchbuffer *batch;
   const struct gen_device_info *devinfo;
};

static void
batchbuffer_init(struct intel_batchbuffer *batch, dri_bufmgr *bufmgr)
{
   memset(batch, 0, sizeof(*batch));

   batch->bo =
      drm_intel_bo_alloc(bufmgr,
                         "batchbuffer brw_can_do_pipelined_register_writes",
                         BATCH_SZ, 4096);
   drm_intel_bo_map(batch->bo, true);
   batch->map = batch->bo->virtual;
   batch->map_next = batch->map;

   batch->reserved_space = BATCH_RESERVED;
   batch->state_batch_offset = batch->bo->size;

   batch->ring = RENDER_RING;
}

static uint32_t
batchbuffer_reloc(struct intel_batchbuffer *batch,
                  drm_intel_bo *buffer, uint32_t offset,
                  uint32_t read_domains, uint32_t write_domain,
                  uint32_t delta)
{
   int ret;

   ret = drm_intel_bo_emit_reloc(batch->bo, offset, buffer, delta,
                                 read_domains, write_domain);
   assert(ret == 0);
   (void)ret;

   /* Using the old buffer offset, write in what the right data would be, in
    * case the buffer doesn't move and we can short-circuit the relocation
    * processing in the kernel
    */
   return buffer->offset64 + delta;
}

static void
emit_mi_flush(struct check_register_writes_context *ctx)
{
   assert(ctx->batch->ring == RENDER_RING);
   assert(ctx->devinfo->gen == 7);

   __BEGIN_BATCH(5);
   __OUT_BATCH(_3DSTATE_PIPE_CONTROL | (5 - 2));
   __OUT_BATCH(0); /* flags */
   __OUT_BATCH(0);
   __OUT_BATCH(0);
   __OUT_BATCH(0);
   __ADVANCE_BATCH();
}

static inline void
batchbuffer_emit_dword(struct intel_batchbuffer *batch, GLuint dword)
{
   *batch->map_next++ = dword;
}

static int
do_flush_locked(struct check_register_writes_context *ctx)
{
   drm_intel_bo_unmap(ctx->batch->bo);
   int ret = drm_intel_gem_bo_context_exec(ctx->batch->bo, ctx->hw_ctx,
                                           4 * USED_BATCH(*(ctx->batch)),
                                           I915_EXEC_RENDER);
   return ret;
}

static int
batchbuffer_flush(struct check_register_writes_context *ctx)
{
   ctx->batch->reserved_space = 0;

   /* Mark the end of the buffer. */
   batchbuffer_emit_dword(ctx->batch, MI_BATCH_BUFFER_END);
   if (USED_BATCH(*(ctx->batch)) & 1) {
      /* Round batchbuffer usage to 2 DWORDs. */
      batchbuffer_emit_dword(ctx->batch, MI_NOOP);
   }

   int ret = do_flush_locked(ctx);

   return ret;
}

static void
batchbuffer_free(struct intel_batchbuffer *batch)
{
   free(batch->cpu_map);
   drm_intel_bo_unreference(batch->last_bo);
   drm_intel_bo_unreference(batch->bo);
   free(batch);
}

static struct check_register_writes_context *
create_ctx(__DRIscreen *dri_screen)
{
   struct check_register_writes_context *ctx =
      (struct check_register_writes_context *)
         malloc(sizeof(struct check_register_writes_context));

   ctx->bufmgr = intel_bufmgr_gem_init(dri_screen->fd, BATCH_SZ);

   ctx->hw_ctx = drm_intel_gem_context_create(ctx->bufmgr);

   ctx->batch =
      (struct intel_batchbuffer *) malloc(sizeof(struct intel_batchbuffer));

   batchbuffer_init(ctx->batch, ctx->bufmgr);

   struct intel_screen *screen =
      (struct intel_screen *) dri_screen->driverPrivate;
   ctx->devinfo = (const struct gen_device_info *) &screen->devinfo;

   return ctx;
}

static void
free_ctx(struct check_register_writes_context *ctx)
{
   batchbuffer_free(ctx->batch);
   drm_intel_gem_context_destroy(ctx->hw_ctx);
   dri_bufmgr_destroy(ctx->bufmgr);
   free(ctx);
}

/*
 * Test if we can use MI_LOAD_REGISTER_MEM from an untrusted batchbuffer.
 *
 * Some combinations of hardware and kernel versions allow this feature,
 * while others don't.  Instead of trying to enumerate every case, just
 * try and write a register and see if works.
 */
int
brw_can_do_pipelined_register_writes(__DRIscreen *dri_screen)
{
   /* gen >= 8 specifically allows these writes. gen <= 6 also
    * doesn't block them.
    */
   struct intel_screen *screen =
      (struct intel_screen *) dri_screen->driverPrivate;
   if (screen->devinfo.gen != 7)
      return true;

   static int result = -1;
   if (result != -1)
      return result;

   struct check_register_writes_context *ctx = create_ctx(dri_screen);

   /* We use SO_WRITE_OFFSET0 since you're supposed to write it (unlike the
    * statistics registers), and we already reset it to zero before using it.
    */
   const int reg = GEN7_SO_WRITE_OFFSET(0);
   const int expected_value = 0x1337d0d0;
   const int offset = 100;

   /* The register we picked only exists on Gen7+. */
   assert(screen->devinfo.gen == 7);

   /* Set a value in a BO to a known quantity */
   uint32_t *data;
   drm_intel_bo *bo =
      drm_intel_bo_alloc(ctx->bufmgr, "brw_can_do_pipelined_register_writes",
                         4096, 4096);

   drm_intel_bo_map(bo, true);
   data = bo->virtual;
   data[offset] = 0xffffffff;
   drm_intel_bo_unmap(bo);

   /* Write the register. */
   __BEGIN_BATCH(3);
   __OUT_BATCH(MI_LOAD_REGISTER_IMM | (3 - 2));
   __OUT_BATCH(reg);
   __OUT_BATCH(expected_value);
   __ADVANCE_BATCH();

   emit_mi_flush(ctx);

   /* Save the register's value back to the buffer. */
   __BEGIN_BATCH(3);
   __OUT_BATCH(MI_STORE_REGISTER_MEM | (3 - 2));
   __OUT_BATCH(reg);
   __OUT_RELOC(bo, offset * sizeof(uint32_t));
   __ADVANCE_BATCH();

   batchbuffer_flush(ctx);

   /* Check whether the value got written. */
   drm_intel_bo_map(bo, false);
   data = bo->virtual;
   bool success = data[offset] == expected_value;
   drm_intel_bo_unmap(bo);

   result = success;

   /* Cleanup */
   drm_intel_bo_unreference(bo);
   free_ctx(ctx);

   return success;
}

#undef __BEGIN_BATCH
#undef __OUT_BATCH
#undef __OUT_RELOC
#undef __ADVANCE_BATCH
