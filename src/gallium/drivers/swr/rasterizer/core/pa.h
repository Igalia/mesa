/****************************************************************************
* Copyright (C) 2014-2015 Intel Corporation.   All Rights Reserved.
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
*
* @file pa.h
*
* @brief Definitions for primitive assembly.
*        N primitives are assembled at a time, where N is the SIMD width.
*        A state machine, that is specific for a given topology, drives the
*        assembly of vertices into triangles.
*
******************************************************************************/
#pragma once

#include "frontend.h"

struct PA_STATE
{
#if USE_SIMD16_FRONTEND
    enum
    {
        SIMD_WIDTH      = KNOB_SIMD16_WIDTH,
        SIMD_WIDTH_DIV2 = KNOB_SIMD16_WIDTH / 2,
        SIMD_WIDTH_LOG2 = 4
    };

    typedef         simd16mask          SIMDMASK;

    typedef         simd16scalar        SIMDSCALAR;
    typedef         simd16vector        SIMDVECTOR;
    typedef         simd16vertex        SIMDVERTEX;

    typedef         simd16scalari       SIMDSCALARI;

#else
    enum
    {
        SIMD_WIDTH      = KNOB_SIMD_WIDTH,
        SIMD_WIDTH_DIV2 = KNOB_SIMD_WIDTH / 2,
        SIMD_WIDTH_LOG2 = 3
    };

    typedef         simdmask            SIMDMASK;

    typedef         simdscalar          SIMDSCALAR;
    typedef         simdvector          SIMDVECTOR;
    typedef         simdvertex          SIMDVERTEX;

    typedef         simdscalari         SIMDSCALARI;

#endif
    DRAW_CONTEXT *pDC{ nullptr };              // draw context
    uint8_t* pStreamBase{ nullptr };           // vertex stream
    uint32_t streamSizeInVerts{ 0 };     // total size of the input stream in verts

    // The topology the binner will use. In some cases the FE changes the topology from the api state.
    PRIMITIVE_TOPOLOGY binTopology{ TOP_UNKNOWN };

#if ENABLE_AVX512_SIMD16
    bool useAlternateOffset{ false };

#endif
    PA_STATE() {}
    PA_STATE(DRAW_CONTEXT *in_pDC, uint8_t* in_pStreamBase, uint32_t in_streamSizeInVerts) :
        pDC(in_pDC), pStreamBase(in_pStreamBase), streamSizeInVerts(in_streamSizeInVerts) {}

    virtual bool HasWork() = 0;
    virtual simdvector& GetSimdVector(uint32_t index, uint32_t slot) = 0;
#if ENABLE_AVX512_SIMD16
    virtual simd16vector& GetSimdVector_simd16(uint32_t index, uint32_t slot) = 0;
#endif
    virtual bool Assemble(uint32_t slot, simdvector verts[]) = 0;
#if ENABLE_AVX512_SIMD16
    virtual bool Assemble_simd16(uint32_t slot, simd16vector verts[]) = 0;
#endif
    virtual void AssembleSingle(uint32_t slot, uint32_t primIndex, __m128 verts[]) = 0;
    virtual bool NextPrim() = 0;
    virtual SIMDVERTEX& GetNextVsOutput() = 0;
    virtual bool GetNextStreamOutput() = 0;
    virtual SIMDMASK& GetNextVsIndices() = 0;
    virtual uint32_t NumPrims() = 0;
    virtual void Reset() = 0;
    virtual SIMDSCALARI GetPrimID(uint32_t startID) = 0;
};

// The Optimized PA is a state machine that assembles triangles from vertex shader simd
// output. Here is the sequence
//    1. Execute FS/VS to generate a simd vertex (4 vertices for SSE simd and 8 for AVX simd).
//    2. Execute PA function to assemble and bin triangles.
//        a.    The PA function is a set of functions that collectively make up the
//            state machine for a given topology.
//                1.    We use a state index to track which PA function to call.
//        b. Often the PA function needs to 2 simd vertices in order to assemble the next triangle.
//                1.    We call this the current and previous simd vertex.
//                2.    The SSE simd is 4-wide which is not a multiple of 3 needed for triangles. In
//                    order to assemble the second triangle, for a triangle list, we'll need the
//                    last vertex from the previous simd and the first 2 vertices from the current simd.
//                3. At times the PA can assemble multiple triangles from the 2 simd vertices.
//
// This optimized PA is not cut aware, so only should be used by non-indexed draws or draws without
// cuts
struct PA_STATE_OPT : public PA_STATE
{
    SIMDVERTEX leadingVertex;            // For tri-fan

    uint32_t numPrims{ 0 };              // Total number of primitives for draw.
    uint32_t numPrimsComplete{ 0 };      // Total number of complete primitives.

    uint32_t numSimdPrims{ 0 };          // Number of prims in current simd.

    uint32_t cur{ 0 };                   // index to current VS output.
    uint32_t prev{ 0 };                  // index to prev VS output. Not really needed in the state.
    uint32_t first{ 0 };                 // index to first VS output. Used for trifan.

    uint32_t counter{ 0 };               // state counter
    bool reset{ false };                 // reset state

    uint32_t primIDIncr{ 0 };            // how much to increment for each vector (typically vector / {1, 2})
    SIMDSCALARI primID;

    typedef bool(*PFN_PA_FUNC)(PA_STATE_OPT& state, uint32_t slot, simdvector verts[]);
#if ENABLE_AVX512_SIMD16
    typedef bool(*PFN_PA_FUNC_SIMD16)(PA_STATE_OPT& state, uint32_t slot, simd16vector verts[]);
#endif
    typedef void(*PFN_PA_SINGLE_FUNC)(PA_STATE_OPT& pa, uint32_t slot, uint32_t primIndex, __m128 verts[]);

    PFN_PA_FUNC        pfnPaFunc{ nullptr };        // PA state machine function for assembling 4 triangles.
#if ENABLE_AVX512_SIMD16
    PFN_PA_FUNC_SIMD16 pfnPaFunc_simd16{ nullptr };
#endif
    PFN_PA_SINGLE_FUNC pfnPaSingleFunc{ nullptr };  // PA state machine function for assembling single triangle.
    PFN_PA_FUNC        pfnPaFuncReset{ nullptr };   // initial state to set on reset
#if ENABLE_AVX512_SIMD16
    PFN_PA_FUNC_SIMD16 pfnPaFuncReset_simd16{ nullptr };
#endif

    // state used to advance the PA when Next is called
    PFN_PA_FUNC        pfnPaNextFunc{ nullptr };
#if ENABLE_AVX512_SIMD16
    PFN_PA_FUNC_SIMD16 pfnPaNextFunc_simd16{ nullptr };
#endif
    uint32_t           nextNumSimdPrims{ 0 };
    uint32_t           nextNumPrimsIncrement{ 0 };
    bool               nextReset{ false };
    bool               isStreaming{ false };

    SIMDMASK tmpIndices{ 0 };            // temporary index store for unused virtual function
    
    PA_STATE_OPT() {}
    PA_STATE_OPT(DRAW_CONTEXT* pDC, uint32_t numPrims, uint8_t* pStream, uint32_t streamSizeInVerts,
        bool in_isStreaming, PRIMITIVE_TOPOLOGY topo = TOP_UNKNOWN);

    bool HasWork()
    {
        return (this->numPrimsComplete < this->numPrims) ? true : false;
    }

    simdvector& GetSimdVector(uint32_t index, uint32_t slot)
    {
        simdvertex* pVertex = (simdvertex*)pStreamBase;
        return pVertex[index].attrib[slot];
    }

#if ENABLE_AVX512_SIMD16
    simd16vector& GetSimdVector_simd16(uint32_t index, uint32_t slot)
    {
        simd16vertex* pVertex = (simd16vertex*)pStreamBase;
        return pVertex[index].attrib[slot];
    }

#endif
    // Assembles 4 triangles. Each simdvector is a single vertex from 4
    // triangles (xxxx yyyy zzzz wwww) and there are 3 verts per triangle.
    bool Assemble(uint32_t slot, simdvector verts[])
    {
        return this->pfnPaFunc(*this, slot, verts);
    }

#if ENABLE_AVX512_SIMD16
    bool Assemble_simd16(uint32_t slot, simd16vector verts[])
    {
        return this->pfnPaFunc_simd16(*this, slot, verts);
    }

#endif
    // Assembles 1 primitive. Each simdscalar is a vertex (xyzw).
    void AssembleSingle(uint32_t slot, uint32_t primIndex, __m128 verts[])
    {
        return this->pfnPaSingleFunc(*this, slot, primIndex, verts);
    }

    bool NextPrim()
    {
        this->pfnPaFunc = this->pfnPaNextFunc;
#if ENABLE_AVX512_SIMD16
        this->pfnPaFunc_simd16 = this->pfnPaNextFunc_simd16;
#endif
        this->numSimdPrims = this->nextNumSimdPrims;
        this->numPrimsComplete += this->nextNumPrimsIncrement;
        this->reset = this->nextReset;

        if (this->isStreaming)
        {
            this->reset = false;
        }

        bool morePrims = false;

        if (this->numSimdPrims > 0)
        {
            morePrims = true;
            this->numSimdPrims--;
        }
        else
        {
            this->counter = (this->reset) ? 0 : (this->counter + 1);
            this->reset = false;
        }

        this->pfnPaFunc = this->pfnPaNextFunc;

        if (!HasWork())
        {
            morePrims = false;    // no more to do
        }

        return morePrims;
    }

    SIMDVERTEX& GetNextVsOutput()
    {
        // increment cur and prev indices
        const uint32_t numSimdVerts = this->streamSizeInVerts / SIMD_WIDTH;
        this->prev = this->cur;  // prev is undefined for first state.
        this->cur = this->counter % numSimdVerts;

        SIMDVERTEX* pVertex = (SIMDVERTEX*)pStreamBase;
        return pVertex[this->cur];
    }

    SIMDMASK& GetNextVsIndices()
    {
        // unused in optimized PA, pass tmp buffer back
        return tmpIndices;
    }

    bool GetNextStreamOutput()
    {
        this->prev = this->cur;
        this->cur = this->counter;

        return HasWork();
    }

    uint32_t NumPrims()
    {
        return (this->numPrimsComplete + this->nextNumPrimsIncrement > this->numPrims) ?
            (SIMD_WIDTH - (this->numPrimsComplete + this->nextNumPrimsIncrement - this->numPrims)) : SIMD_WIDTH;
    }

    void SetNextState(PA_STATE_OPT::PFN_PA_FUNC pfnPaNextFunc,
        PA_STATE_OPT::PFN_PA_SINGLE_FUNC pfnPaNextSingleFunc,
        uint32_t numSimdPrims = 0,
        uint32_t numPrimsIncrement = 0,
        bool reset = false)
    {
        this->pfnPaNextFunc = pfnPaNextFunc;
        this->nextNumSimdPrims = numSimdPrims;
        this->nextNumPrimsIncrement = numPrimsIncrement;
        this->nextReset = reset;

        this->pfnPaSingleFunc = pfnPaNextSingleFunc;
    }

#if ENABLE_AVX512_SIMD16
    void SetNextState_simd16(PA_STATE_OPT::PFN_PA_FUNC_SIMD16 pfnPaNextFunc_simd16,
        PA_STATE_OPT::PFN_PA_SINGLE_FUNC pfnPaNextSingleFunc,
        uint32_t numSimdPrims = 0,
        uint32_t numPrimsIncrement = 0,
        bool reset = false)
    {
        this->pfnPaNextFunc_simd16 = pfnPaNextFunc_simd16;
        this->nextNumSimdPrims = numSimdPrims;
        this->nextNumPrimsIncrement = numPrimsIncrement;
        this->nextReset = reset;

        this->pfnPaSingleFunc = pfnPaNextSingleFunc;
    }

#endif
    void Reset()
    {
#if ENABLE_AVX512_SIMD16
        useAlternateOffset = false;

#endif
        this->pfnPaFunc = this->pfnPaFuncReset;
        this->numPrimsComplete = 0;
        this->numSimdPrims = 0;
        this->cur = 0;
        this->prev = 0;
        this->first = 0;
        this->counter = 0;
        this->reset = false;
    }

    SIMDSCALARI GetPrimID(uint32_t startID)
    {
#if USE_SIMD16_FRONTEND
        return _simd16_add_epi32(this->primID,
            _simd16_set1_epi32(startID + this->primIDIncr * (this->numPrimsComplete / SIMD_WIDTH)));
#else
        return _simd_add_epi32(this->primID,
            _simd_set1_epi32(startID + this->primIDIncr * (this->numPrimsComplete / SIMD_WIDTH)));
#endif
    }
};

// helper C wrappers to avoid having to rewrite all the PA topology state functions
INLINE void SetNextPaState(PA_STATE_OPT& pa, PA_STATE_OPT::PFN_PA_FUNC pfnPaNextFunc,
    PA_STATE_OPT::PFN_PA_SINGLE_FUNC pfnPaNextSingleFunc,
    uint32_t numSimdPrims = 0,
    uint32_t numPrimsIncrement = 0,
    bool reset = false)
{
    return pa.SetNextState(pfnPaNextFunc, pfnPaNextSingleFunc, numSimdPrims, numPrimsIncrement, reset);
}

#if ENABLE_AVX512_SIMD16
INLINE void SetNextPaState_simd16(PA_STATE_OPT& pa, PA_STATE_OPT::PFN_PA_FUNC_SIMD16 pfnPaNextFunc_simd16,
    PA_STATE_OPT::PFN_PA_SINGLE_FUNC pfnPaNextSingleFunc,
    uint32_t numSimdPrims = 0,
    uint32_t numPrimsIncrement = 0,
    bool reset = false)
{
    return pa.SetNextState_simd16(pfnPaNextFunc_simd16, pfnPaNextSingleFunc, numSimdPrims, numPrimsIncrement, reset);
}

#endif
INLINE simdvector& PaGetSimdVector(PA_STATE& pa, uint32_t index, uint32_t slot)
{
    return pa.GetSimdVector(index, slot);
}

#if ENABLE_AVX512_SIMD16
INLINE simd16vector& PaGetSimdVector_simd16(PA_STATE& pa, uint32_t index, uint32_t slot)
{
    return pa.GetSimdVector_simd16(index, slot);
}

#endif
INLINE __m128 swizzleLane0(const simdvector &a)
{
    simdscalar tmp0 = _mm256_unpacklo_ps(a.x, a.z);
    simdscalar tmp1 = _mm256_unpacklo_ps(a.y, a.w);
    return _mm256_extractf128_ps(_mm256_unpacklo_ps(tmp0, tmp1), 0);
}

INLINE __m128 swizzleLane1(const simdvector &a)
{
    simdscalar tmp0 = _mm256_unpacklo_ps(a.x, a.z);
    simdscalar tmp1 = _mm256_unpacklo_ps(a.y, a.w);
    return _mm256_extractf128_ps(_mm256_unpackhi_ps(tmp0, tmp1), 0);
}

INLINE __m128 swizzleLane2(const simdvector &a)
{
    simdscalar tmp0 = _mm256_unpackhi_ps(a.x, a.z);
    simdscalar tmp1 = _mm256_unpackhi_ps(a.y, a.w);
    return _mm256_extractf128_ps(_mm256_unpacklo_ps(tmp0, tmp1), 0);
}

INLINE __m128 swizzleLane3(const simdvector &a)
{
    simdscalar tmp0 = _mm256_unpackhi_ps(a.x, a.z);
    simdscalar tmp1 = _mm256_unpackhi_ps(a.y, a.w);
    return _mm256_extractf128_ps(_mm256_unpackhi_ps(tmp0, tmp1), 0);
}

INLINE __m128 swizzleLane4(const simdvector &a)
{
    simdscalar tmp0 = _mm256_unpacklo_ps(a.x, a.z);
    simdscalar tmp1 = _mm256_unpacklo_ps(a.y, a.w);
    return _mm256_extractf128_ps(_mm256_unpacklo_ps(tmp0, tmp1), 1);

}

INLINE __m128 swizzleLane5(const simdvector &a)
{
    simdscalar tmp0 = _mm256_unpacklo_ps(a.x, a.z);
    simdscalar tmp1 = _mm256_unpacklo_ps(a.y, a.w);
    return _mm256_extractf128_ps(_mm256_unpackhi_ps(tmp0, tmp1), 1);
}

INLINE __m128 swizzleLane6(const simdvector &a)
{
    simdscalar tmp0 = _mm256_unpackhi_ps(a.x, a.z);
    simdscalar tmp1 = _mm256_unpackhi_ps(a.y, a.w);
    return _mm256_extractf128_ps(_mm256_unpacklo_ps(tmp0, tmp1), 1);
}

INLINE __m128 swizzleLane7(const simdvector &a)
{
    simdscalar tmp0 = _mm256_unpackhi_ps(a.x, a.z);
    simdscalar tmp1 = _mm256_unpackhi_ps(a.y, a.w);
    return _mm256_extractf128_ps(_mm256_unpackhi_ps(tmp0, tmp1), 1);
}

INLINE __m128 swizzleLaneN(const simdvector &a, int lane)
{
    switch (lane) {
    case 0:
        return swizzleLane0(a);
    case 1:
        return swizzleLane1(a);
    case 2:
        return swizzleLane2(a);
    case 3:
        return swizzleLane3(a);
    case 4:
        return swizzleLane4(a);
    case 5:
        return swizzleLane5(a);
    case 6:
        return swizzleLane6(a);
    case 7:
        return swizzleLane7(a);
    default:
        return _mm_setzero_ps();
    }
}

// Cut-aware primitive assembler.
struct PA_STATE_CUT : public PA_STATE
{
    SIMDMASK* pCutIndices{ nullptr };    // cut indices buffer, 1 bit per vertex
    uint32_t numVerts{ 0 };              // number of vertices available in buffer store
    uint32_t numAttribs{ 0 };            // number of attributes
    int32_t numRemainingVerts{ 0 };      // number of verts remaining to be assembled
    uint32_t numVertsToAssemble{ 0 };    // total number of verts to assemble for the draw
#if ENABLE_AVX512_SIMD16
    OSALIGNSIMD16(uint32_t) indices[MAX_NUM_VERTS_PER_PRIM][SIMD_WIDTH];    // current index buffer for gather
#else
    OSALIGNSIMD(uint32_t) indices[MAX_NUM_VERTS_PER_PRIM][SIMD_WIDTH];    // current index buffer for gather
#endif
    SIMDSCALARI vOffsets[MAX_NUM_VERTS_PER_PRIM];           // byte offsets for currently assembling simd
    uint32_t numPrimsAssembled{ 0 };     // number of primitives that are fully assembled
    uint32_t headVertex{ 0 };            // current unused vertex slot in vertex buffer store
    uint32_t tailVertex{ 0 };            // beginning vertex currently assembling
    uint32_t curVertex{ 0 };             // current unprocessed vertex
    uint32_t startPrimId{ 0 };           // starting prim id
    SIMDSCALARI vPrimId;                 // vector of prim ID
    bool needOffsets{ false };           // need to compute gather offsets for current SIMD
    uint32_t vertsPerPrim{ 0 };
    SIMDVERTEX tmpVertex;                // temporary simdvertex for unimplemented API
    bool processCutVerts{ false };       // vertex indices with cuts should be processed as normal, otherwise they
                                         // are ignored.  Fetch shader sends invalid verts on cuts that should be ignored
                                         // while the GS sends valid verts for every index 
    // Topology state tracking
    uint32_t vert[MAX_NUM_VERTS_PER_PRIM];
    uint32_t curIndex{ 0 };
    bool reverseWinding{ false };        // indicates reverse winding for strips
    int32_t adjExtraVert{ 0 };           // extra vert uses for tristrip w/ adj

    typedef void(PA_STATE_CUT::* PFN_PA_FUNC)(uint32_t vert, bool finish);
    PFN_PA_FUNC pfnPa{ nullptr };        // per-topology function that processes a single vert

    PA_STATE_CUT() {}
    PA_STATE_CUT(DRAW_CONTEXT* pDC, uint8_t* in_pStream, uint32_t in_streamSizeInVerts, SIMDMASK* in_pIndices, uint32_t in_numVerts,
        uint32_t in_numAttribs, PRIMITIVE_TOPOLOGY topo, bool in_processCutVerts)
        : PA_STATE(pDC, in_pStream, in_streamSizeInVerts)
    {
        numVerts = in_streamSizeInVerts;
        numAttribs = in_numAttribs;
        binTopology = topo;
        needOffsets = false;
        processCutVerts = in_processCutVerts;

        numVertsToAssemble = numRemainingVerts = in_numVerts;
        numPrimsAssembled = 0;
        headVertex = tailVertex = curVertex = 0;

        curIndex = 0;
        pCutIndices = in_pIndices;
        memset(indices, 0, sizeof(indices));
#if USE_SIMD16_FRONTEND
        vPrimId = _simd16_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
#else
        vPrimId = _simd_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
#endif
        reverseWinding = false;
        adjExtraVert = -1;

        bool gsEnabled = pDC->pState->state.gsState.gsEnable;
        vertsPerPrim = NumVertsPerPrim(topo, gsEnabled);

        switch (topo)
        {
        case TOP_TRIANGLE_LIST:     pfnPa = &PA_STATE_CUT::ProcessVertTriList; break;
        case TOP_TRI_LIST_ADJ:      pfnPa = gsEnabled ? &PA_STATE_CUT::ProcessVertTriListAdj : &PA_STATE_CUT::ProcessVertTriListAdjNoGs; break;
        case TOP_TRIANGLE_STRIP:    pfnPa = &PA_STATE_CUT::ProcessVertTriStrip; break;
        case TOP_TRI_STRIP_ADJ:     if (gsEnabled)
                                    {
                                        pfnPa = &PA_STATE_CUT::ProcessVertTriStripAdj < true > ;
                                    }
                                    else
                                    {
                                        pfnPa = &PA_STATE_CUT::ProcessVertTriStripAdj < false > ;
                                    }
                                    break;

        case TOP_POINT_LIST:        pfnPa = &PA_STATE_CUT::ProcessVertPointList; break;
        case TOP_LINE_LIST:         pfnPa = &PA_STATE_CUT::ProcessVertLineList; break;
        case TOP_LINE_LIST_ADJ:     pfnPa = gsEnabled ? &PA_STATE_CUT::ProcessVertLineListAdj : &PA_STATE_CUT::ProcessVertLineListAdjNoGs; break;
        case TOP_LINE_STRIP:        pfnPa = &PA_STATE_CUT::ProcessVertLineStrip; break;
        case TOP_LISTSTRIP_ADJ:     pfnPa = gsEnabled ? &PA_STATE_CUT::ProcessVertLineStripAdj : &PA_STATE_CUT::ProcessVertLineStripAdjNoGs; break;
        default: assert(0 && "Unimplemented topology");
        }
    }

    SIMDVERTEX& GetNextVsOutput()
    {
        uint32_t vertexIndex = this->headVertex / SIMD_WIDTH;
        this->headVertex = (this->headVertex + SIMD_WIDTH) % this->numVerts;
        this->needOffsets = true;
        return ((SIMDVERTEX*)pStreamBase)[vertexIndex];
    }

    SIMDMASK& GetNextVsIndices()
    {
        uint32_t vertexIndex = this->headVertex / SIMD_WIDTH;
        SIMDMASK* pCurCutIndex = this->pCutIndices + vertexIndex;
        return *pCurCutIndex;
    }

    simdvector& GetSimdVector(uint32_t index, uint32_t slot)
    {
        // unused
        SWR_ASSERT(0 && "Not implemented");
        static simdvector junk;
        return junk;
    }

#if ENABLE_AVX512_SIMD16
    simd16vector& GetSimdVector_simd16(uint32_t index, uint32_t slot)
    {
        // unused
        SWR_ASSERT(0 && "Not implemented");
        static simd16vector junk;
        return junk;
    }

#endif
    bool GetNextStreamOutput()
    {
        this->headVertex += SIMD_WIDTH;
        this->needOffsets = true;
        return HasWork();
    }

    SIMDSCALARI GetPrimID(uint32_t startID)
    {
#if USE_SIMD16_FRONTEND
        return _simd16_add_epi32(_simd16_set1_epi32(startID), this->vPrimId);
#else
        return _simd_add_epi32(_simd_set1_epi32(startID), this->vPrimId);
#endif
    }

    void Reset()
    {
#if ENABLE_AVX512_SIMD16
        useAlternateOffset = false;

#endif
        this->numRemainingVerts = this->numVertsToAssemble;
        this->numPrimsAssembled = 0;
        this->curIndex = 0;
        this->curVertex = 0;
        this->tailVertex = 0;
        this->headVertex = 0;
        this->reverseWinding = false;
        this->adjExtraVert = -1;
#if USE_SIMD16_FRONTEND
        this->vPrimId = _simd16_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
#else
        this->vPrimId = _simd_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
#endif
    }

    bool HasWork()
    {
        return this->numRemainingVerts > 0 || this->adjExtraVert != -1;
    }

    bool IsVertexStoreFull()
    {
        return ((this->headVertex + SIMD_WIDTH) % this->numVerts) == this->tailVertex;
    }

    void RestartTopology()
    {
        this->curIndex = 0;
        this->reverseWinding = false;
        this->adjExtraVert = -1;
    }

    bool IsCutIndex(uint32_t vertex)
    {
        uint32_t vertexIndex = vertex / SIMD_WIDTH;
        uint32_t vertexOffset = vertex & (SIMD_WIDTH - 1);
        return _bittest((const LONG*)&this->pCutIndices[vertexIndex], vertexOffset) == 1;
    }

    // iterates across the unprocessed verts until we hit the end or we 
    // have assembled SIMD prims
    void ProcessVerts()
    {
        while (this->numPrimsAssembled != SIMD_WIDTH &&
            this->numRemainingVerts > 0 &&
            this->curVertex != this->headVertex)
        {
            // if cut index, restart topology 
            if (IsCutIndex(this->curVertex))
            {
                if (this->processCutVerts)
                {
                    (this->*pfnPa)(this->curVertex, false);
                }
                // finish off tri strip w/ adj before restarting topo
                if (this->adjExtraVert != -1)
                {
                    (this->*pfnPa)(this->curVertex, true);
                }
                RestartTopology();
            }
            else
            {
                (this->*pfnPa)(this->curVertex, false);
            }

            this->curVertex++;
            if (this->curVertex >= this->numVerts) {
               this->curVertex = 0;
            }
            this->numRemainingVerts--;
        }

        // special case last primitive for tri strip w/ adj
        if (this->numPrimsAssembled != SIMD_WIDTH && this->numRemainingVerts == 0 && this->adjExtraVert != -1)
        {
            (this->*pfnPa)(this->curVertex, true);
        }
    }

    void Advance()
    {
        // done with current batch
        // advance tail to the current unsubmitted vertex
        this->tailVertex = this->curVertex;
        this->numPrimsAssembled = 0;
#if USE_SIMD16_FRONTEND
        this->vPrimId = _simd16_add_epi32(vPrimId, _simd16_set1_epi32(SIMD_WIDTH));
#else
        this->vPrimId = _simd_add_epi32(vPrimId, _simd_set1_epi32(SIMD_WIDTH));
#endif
    }

    bool NextPrim()
    {
        // if we've assembled enough prims, we can advance to the next set of verts
        if (this->numPrimsAssembled == SIMD_WIDTH || this->numRemainingVerts <= 0)
        {
            Advance();
        }
        return false;
    }

    void ComputeOffsets()
    {
        for (uint32_t v = 0; v < this->vertsPerPrim; ++v)
        {
            SIMDSCALARI vIndices = *(SIMDSCALARI*)&this->indices[v][0];

            // step to simdvertex batch
            const uint32_t simdShift = SIMD_WIDTH_LOG2;
#if USE_SIMD16_FRONTEND
            SIMDSCALARI vVertexBatch = _simd16_srai_epi32(vIndices, simdShift);
            this->vOffsets[v] = _simd16_mullo_epi32(vVertexBatch, _simd16_set1_epi32(sizeof(SIMDVERTEX)));
#else
            SIMDSCALARI vVertexBatch = _simd_srai_epi32(vIndices, simdShift);
            this->vOffsets[v] = _simd_mullo_epi32(vVertexBatch, _simd_set1_epi32(sizeof(SIMDVERTEX)));
#endif

            // step to index
            const uint32_t simdMask = SIMD_WIDTH - 1;
#if USE_SIMD16_FRONTEND
            SIMDSCALARI vVertexIndex = _simd16_and_si(vIndices, _simd16_set1_epi32(simdMask));
            this->vOffsets[v] = _simd16_add_epi32(this->vOffsets[v], _simd16_mullo_epi32(vVertexIndex, _simd16_set1_epi32(sizeof(float))));
#else
            SIMDSCALARI vVertexIndex = _simd_and_si(vIndices, _simd_set1_epi32(simdMask));
            this->vOffsets[v] = _simd_add_epi32(this->vOffsets[v], _simd_mullo_epi32(vVertexIndex, _simd_set1_epi32(sizeof(float))));
#endif
        }
    }

    bool Assemble(uint32_t slot, simdvector verts[])
    {
        // process any outstanding verts
        ProcessVerts();

        // return false if we don't have enough prims assembled
        if (this->numPrimsAssembled != SIMD_WIDTH && this->numRemainingVerts > 0)
        {
            return false;
        }

        // cache off gather offsets given the current SIMD set of indices the first time we get an assemble
        if (this->needOffsets)
        {
            ComputeOffsets();
            this->needOffsets = false;
        }

        for (uint32_t v = 0; v < this->vertsPerPrim; ++v)
        {
            SIMDSCALARI offsets = this->vOffsets[v];

            // step to attribute
#if USE_SIMD16_FRONTEND
            offsets = _simd16_add_epi32(offsets, _simd16_set1_epi32(slot * sizeof(SIMDVECTOR)));
#else
            offsets = _simd_add_epi32(offsets, _simd_set1_epi32(slot * sizeof(SIMDVECTOR)));
#endif

            float* pBase = (float*)this->pStreamBase;
            for (uint32_t c = 0; c < 4; ++c)
            {
#if USE_SIMD16_FRONTEND
                simd16scalar temp = _simd16_i32gather_ps(pBase, offsets, 1);

                verts[v].v[c] = useAlternateOffset ? temp.hi : temp.lo;
#else
                verts[v].v[c] = _simd_i32gather_ps(pBase, offsets, 1);
#endif

                // move base to next component
                pBase += SIMD_WIDTH;
            }
        }

        return true;
    }

#if ENABLE_AVX512_SIMD16
    bool Assemble_simd16(uint32_t slot, simd16vector verts[])
    {
        // process any outstanding verts
        ProcessVerts();

        // return false if we don't have enough prims assembled
        if (this->numPrimsAssembled != SIMD_WIDTH && this->numRemainingVerts > 0)
        {
            return false;
        }

        // cache off gather offsets given the current SIMD set of indices the first time we get an assemble
        if (this->needOffsets)
        {
            ComputeOffsets();
            this->needOffsets = false;
        }

        for (uint32_t v = 0; v < this->vertsPerPrim; ++v)
        {
            SIMDSCALARI offsets = this->vOffsets[v];

            // step to attribute
#if USE_SIMD16_FRONTEND
            offsets = _simd16_add_epi32(offsets, _simd16_set1_epi32(slot * sizeof(SIMDVECTOR)));
#else
            offsets = _simd_add_epi32(offsets, _simd_set1_epi32(slot * sizeof(simdvector)));
#endif

            float* pBase = (float*)this->pStreamBase;
            for (uint32_t c = 0; c < 4; ++c)
            {
#if USE_SIMD16_FRONTEND
                verts[v].v[c] = _simd16_i32gather_ps(pBase, offsets, 1);
#else
                verts[v].v[c].lo = _simd_i32gather_ps(pBase, offsets, 1);
                verts[v].v[c].hi = _simd_setzero_ps();
#endif

                // move base to next component
                pBase += SIMD_WIDTH;
            }
        }

        return true;
    }

#endif
    void AssembleSingle(uint32_t slot, uint32_t triIndex, __m128 tri[3])
    {
        // move to slot
        for (uint32_t v = 0; v < this->vertsPerPrim; ++v)
        {
            uint32_t* pOffset = (uint32_t*)&this->vOffsets[v];
#if USE_SIMD16_FRONTEND
            uint32_t offset = useAlternateOffset ? pOffset[triIndex + SIMD_WIDTH_DIV2] : pOffset[triIndex];
#else
            uint32_t offset = pOffset[triIndex];
#endif
            offset += sizeof(SIMDVECTOR) * slot;
            float* pVert = (float*)&tri[v];
            for (uint32_t c = 0; c < 4; ++c)
            {
                float* pComponent = (float*)(this->pStreamBase + offset);
                pVert[c] = *pComponent;
                offset += SIMD_WIDTH * sizeof(float);
            }
        }
    }

    uint32_t NumPrims()
    {
        return this->numPrimsAssembled;
    }

    // Per-topology functions
    void ProcessVertTriStrip(uint32_t index, bool finish)
    {
        this->vert[this->curIndex] = index;
        this->curIndex++;
        if (this->curIndex == 3)
        {
            // assembled enough verts for prim, add to gather indices
            this->indices[0][this->numPrimsAssembled] = this->vert[0];
            if (reverseWinding)
            {
                this->indices[1][this->numPrimsAssembled] = this->vert[2];
                this->indices[2][this->numPrimsAssembled] = this->vert[1];
            }
            else
            {
                this->indices[1][this->numPrimsAssembled] = this->vert[1];
                this->indices[2][this->numPrimsAssembled] = this->vert[2];
            }

            // increment numPrimsAssembled
            this->numPrimsAssembled++;

            // set up next prim state
            this->vert[0] = this->vert[1];
            this->vert[1] = this->vert[2];
            this->curIndex = 2;
            this->reverseWinding ^= 1;
        }
    }

    template<bool gsEnabled>
    void AssembleTriStripAdj()
    {
        if (!gsEnabled)
        {
            this->vert[1] = this->vert[2];
            this->vert[2] = this->vert[4];

            this->indices[0][this->numPrimsAssembled] = this->vert[0];
            this->indices[1][this->numPrimsAssembled] = this->vert[1];
            this->indices[2][this->numPrimsAssembled] = this->vert[2];

            this->vert[4] = this->vert[2];
            this->vert[2] = this->vert[1];
        }
        else
        {
            this->indices[0][this->numPrimsAssembled] = this->vert[0];
            this->indices[1][this->numPrimsAssembled] = this->vert[1];
            this->indices[2][this->numPrimsAssembled] = this->vert[2];
            this->indices[3][this->numPrimsAssembled] = this->vert[3];
            this->indices[4][this->numPrimsAssembled] = this->vert[4];
            this->indices[5][this->numPrimsAssembled] = this->vert[5];
        }
        this->numPrimsAssembled++;
    }


    template<bool gsEnabled>
    void ProcessVertTriStripAdj(uint32_t index, bool finish)
    {
        // handle last primitive of tristrip
        if (finish && this->adjExtraVert != -1)
        {
            this->vert[3] = this->adjExtraVert;
            AssembleTriStripAdj<gsEnabled>();
            this->adjExtraVert = -1;
            return;
        }

        switch (this->curIndex)
        {
        case 0:
        case 1:
        case 2:
        case 4:
            this->vert[this->curIndex] = index;
            this->curIndex++;
            break;
        case 3:
            this->vert[5] = index;
            this->curIndex++;
            break;
        case 5:
            if (this->adjExtraVert == -1)
            {
                this->adjExtraVert = index;
            }
            else
            {
                this->vert[3] = index;
                if (!gsEnabled)
                {
                    AssembleTriStripAdj<gsEnabled>();

                    uint32_t nextTri[6];
                    if (this->reverseWinding)
                    {
                        nextTri[0] = this->vert[4];
                        nextTri[1] = this->vert[0];
                        nextTri[2] = this->vert[2];
                        nextTri[4] = this->vert[3];
                        nextTri[5] = this->adjExtraVert;
                    }
                    else
                    {
                        nextTri[0] = this->vert[2];
                        nextTri[1] = this->adjExtraVert;
                        nextTri[2] = this->vert[3];
                        nextTri[4] = this->vert[4];
                        nextTri[5] = this->vert[0];
                    }
                    for (uint32_t i = 0; i < 6; ++i)
                    {
                        this->vert[i] = nextTri[i];
                    }

                    this->adjExtraVert = -1;
                    this->reverseWinding ^= 1;
                }
                else
                {
                    this->curIndex++;
                }
            }
            break;
        case 6:
            SWR_ASSERT(this->adjExtraVert != -1, "Algorith failure!");
            AssembleTriStripAdj<gsEnabled>();
            
            uint32_t nextTri[6];
            if (this->reverseWinding)
            {
                nextTri[0] = this->vert[4];
                nextTri[1] = this->vert[0];
                nextTri[2] = this->vert[2];
                nextTri[4] = this->vert[3];
                nextTri[5] = this->adjExtraVert;
            }
            else
            {
                nextTri[0] = this->vert[2];
                nextTri[1] = this->adjExtraVert;
                nextTri[2] = this->vert[3];
                nextTri[4] = this->vert[4];
                nextTri[5] = this->vert[0]; 
            }
            for (uint32_t i = 0; i < 6; ++i)
            {
                this->vert[i] = nextTri[i];
            }
            this->reverseWinding ^= 1;
            this->adjExtraVert = index;
            this->curIndex--;
            break;
        }
    }

    void ProcessVertTriList(uint32_t index, bool finish)
    {
        this->vert[this->curIndex] = index;
        this->curIndex++;
        if (this->curIndex == 3)
        {
            // assembled enough verts for prim, add to gather indices
            this->indices[0][this->numPrimsAssembled] = this->vert[0];
            this->indices[1][this->numPrimsAssembled] = this->vert[1];
            this->indices[2][this->numPrimsAssembled] = this->vert[2];

            // increment numPrimsAssembled
            this->numPrimsAssembled++;

            // set up next prim state
            this->curIndex = 0;
        }
    }

    void ProcessVertTriListAdj(uint32_t index, bool finish)
    {
        this->vert[this->curIndex] = index;
        this->curIndex++;
        if (this->curIndex == 6)
        {
            // assembled enough verts for prim, add to gather indices
            this->indices[0][this->numPrimsAssembled] = this->vert[0];
            this->indices[1][this->numPrimsAssembled] = this->vert[1];
            this->indices[2][this->numPrimsAssembled] = this->vert[2];
            this->indices[3][this->numPrimsAssembled] = this->vert[3];
            this->indices[4][this->numPrimsAssembled] = this->vert[4];
            this->indices[5][this->numPrimsAssembled] = this->vert[5];

            // increment numPrimsAssembled
            this->numPrimsAssembled++;

            // set up next prim state
            this->curIndex = 0;
        }
    }

    void ProcessVertTriListAdjNoGs(uint32_t index, bool finish)
    {
        this->vert[this->curIndex] = index;
        this->curIndex++;
        if (this->curIndex == 6)
        {
            // assembled enough verts for prim, add to gather indices
            this->indices[0][this->numPrimsAssembled] = this->vert[0];
            this->indices[1][this->numPrimsAssembled] = this->vert[2];
            this->indices[2][this->numPrimsAssembled] = this->vert[4];

            // increment numPrimsAssembled
            this->numPrimsAssembled++;

            // set up next prim state
            this->curIndex = 0;
        }
    }


    void ProcessVertLineList(uint32_t index, bool finish)
    {
        this->vert[this->curIndex] = index;
        this->curIndex++;
        if (this->curIndex == 2)
        {
            this->indices[0][this->numPrimsAssembled] = this->vert[0];
            this->indices[1][this->numPrimsAssembled] = this->vert[1];

            this->numPrimsAssembled++;
            this->curIndex = 0;
        }
    }

    void ProcessVertLineStrip(uint32_t index, bool finish)
    {
        this->vert[this->curIndex] = index;
        this->curIndex++;
        if (this->curIndex == 2)
        {
            // assembled enough verts for prim, add to gather indices
            this->indices[0][this->numPrimsAssembled] = this->vert[0];
            this->indices[1][this->numPrimsAssembled] = this->vert[1];

            // increment numPrimsAssembled
            this->numPrimsAssembled++;

            // set up next prim state
            this->vert[0] = this->vert[1];
            this->curIndex = 1;
        }
    }

    void ProcessVertLineStripAdj(uint32_t index, bool finish)
    {
        this->vert[this->curIndex] = index;
        this->curIndex++;
        if (this->curIndex == 4)
        {
            // assembled enough verts for prim, add to gather indices
            this->indices[0][this->numPrimsAssembled] = this->vert[0];
            this->indices[1][this->numPrimsAssembled] = this->vert[1];
            this->indices[2][this->numPrimsAssembled] = this->vert[2];
            this->indices[3][this->numPrimsAssembled] = this->vert[3];

            // increment numPrimsAssembled
            this->numPrimsAssembled++;

            // set up next prim state
            this->vert[0] = this->vert[1];
            this->vert[1] = this->vert[2];
            this->vert[2] = this->vert[3];
            this->curIndex = 3;
        }
    }

    void ProcessVertLineStripAdjNoGs(uint32_t index, bool finish)
    {
        this->vert[this->curIndex] = index;
        this->curIndex++;
        if (this->curIndex == 4)
        {
            // assembled enough verts for prim, add to gather indices
            this->indices[0][this->numPrimsAssembled] = this->vert[1];
            this->indices[1][this->numPrimsAssembled] = this->vert[2];

            // increment numPrimsAssembled
            this->numPrimsAssembled++;

            // set up next prim state
            this->vert[0] = this->vert[1];
            this->vert[1] = this->vert[2];
            this->vert[2] = this->vert[3];
            this->curIndex = 3;
        }
    }

    void ProcessVertLineListAdj(uint32_t index, bool finish)
    {
        this->vert[this->curIndex] = index;
        this->curIndex++;
        if (this->curIndex == 4)
        {
            this->indices[0][this->numPrimsAssembled] = this->vert[0];
            this->indices[1][this->numPrimsAssembled] = this->vert[1];
            this->indices[2][this->numPrimsAssembled] = this->vert[2];
            this->indices[3][this->numPrimsAssembled] = this->vert[3];

            this->numPrimsAssembled++;
            this->curIndex = 0;
        }
    }

    void ProcessVertLineListAdjNoGs(uint32_t index, bool finish)
    {
        this->vert[this->curIndex] = index;
        this->curIndex++;
        if (this->curIndex == 4)
        {
            this->indices[0][this->numPrimsAssembled] = this->vert[1];
            this->indices[1][this->numPrimsAssembled] = this->vert[2];

            this->numPrimsAssembled++;
            this->curIndex = 0;
        }
    }

    void ProcessVertPointList(uint32_t index, bool finish)
    {
        this->vert[this->curIndex] = index;
        this->curIndex++;
        if (this->curIndex == 1)
        {
            this->indices[0][this->numPrimsAssembled] = this->vert[0];
            this->numPrimsAssembled++;
            this->curIndex = 0;
        }
    }
};

// Primitive Assembly for data output from the DomainShader.
struct PA_TESS : PA_STATE
{
    PA_TESS(
        DRAW_CONTEXT *in_pDC,
        const SIMDSCALAR* in_pVertData,
        uint32_t in_attributeStrideInVectors,
        uint32_t in_numAttributes,
        uint32_t* (&in_ppIndices)[3],
        uint32_t in_numPrims,
        PRIMITIVE_TOPOLOGY in_binTopology) :

        PA_STATE(in_pDC, nullptr, 0),
        m_pVertexData(in_pVertData),
        m_attributeStrideInVectors(in_attributeStrideInVectors),
        m_numAttributes(in_numAttributes),
        m_numPrims(in_numPrims)
    {
#if USE_SIMD16_FRONTEND
        m_vPrimId = _simd16_setzero_si();
#else
        m_vPrimId = _simd_setzero_si();
#endif
        binTopology = in_binTopology;
        m_ppIndices[0] = in_ppIndices[0];
        m_ppIndices[1] = in_ppIndices[1];
        m_ppIndices[2] = in_ppIndices[2];

        switch (binTopology)
        {
        case TOP_POINT_LIST:
            m_numVertsPerPrim = 1;
            break;

        case TOP_LINE_LIST:
            m_numVertsPerPrim = 2;
            break;

        case TOP_TRIANGLE_LIST:
            m_numVertsPerPrim = 3;
            break;

        default:
            SWR_ASSERT(0, "Invalid binTopology (%d) for %s", binTopology, __FUNCTION__);
            break;
        }
    }

    bool HasWork()
    {
        return m_numPrims != 0;
    }

    simdvector& GetSimdVector(uint32_t index, uint32_t slot)
    {
        SWR_ASSERT(0, "%s NOT IMPLEMENTED", __FUNCTION__);
        static simdvector junk;
        return junk;
    }

#if ENABLE_AVX512_SIMD16
    simd16vector& GetSimdVector_simd16(uint32_t index, uint32_t slot)
    {
        SWR_ASSERT(0, "%s NOT IMPLEMENTED", __FUNCTION__);
        static simd16vector junk;
        return junk;
    }

#endif
    static SIMDSCALARI GenPrimMask(uint32_t numPrims)
    {
        SWR_ASSERT(numPrims <= SIMD_WIDTH);
#if USE_SIMD16_FRONTEND
        static const OSALIGNLINE(int32_t) maskGen[SIMD_WIDTH * 2] =
        {
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
        };

        return _simd16_loadu_si((const SIMDSCALARI*)&maskGen[SIMD_WIDTH - numPrims]);
#else
        static const OSALIGNLINE(int32_t) maskGen[SIMD_WIDTH * 2] =
        {
            -1, -1, -1, -1, -1, -1, -1, -1,
            0,  0,  0,  0,  0,  0,  0,  0
        };

        return _simd_loadu_si((const SIMDSCALARI*)&maskGen[SIMD_WIDTH - numPrims]);
#endif
    }

    bool Assemble(uint32_t slot, simdvector verts[])
    {
        SWR_ASSERT(slot < m_numAttributes);

        uint32_t numPrimsToAssemble = PA_TESS::NumPrims();
        if (0 == numPrimsToAssemble)
        {
            return false;
        }

        SIMDSCALARI mask = GenPrimMask(numPrimsToAssemble);

        const float* pBaseAttrib = (const float*)&m_pVertexData[slot * m_attributeStrideInVectors * 4];
        for (uint32_t i = 0; i < m_numVertsPerPrim; ++i)
        {
#if USE_SIMD16_FRONTEND
            SIMDSCALARI indices = _simd16_load_si((const SIMDSCALARI*)m_ppIndices[i]);
#else
            SIMDSCALARI indices = _simd_load_si((const SIMDSCALARI*)m_ppIndices[i]);
#endif

            const float* pBase = pBaseAttrib;
            for (uint32_t c = 0; c < 4; ++c)
            {
#if USE_SIMD16_FRONTEND
                simd16scalar temp = _simd16_mask_i32gather_ps(
                    _simd16_setzero_ps(),
                    pBase,
                    indices,
                    mask,
                    4 /* gcc doesn't like sizeof(float) */);

                verts[i].v[c] = useAlternateOffset ? temp.hi : temp.lo;
#else
                verts[i].v[c] = _simd_mask_i32gather_ps(
                    _simd_setzero_ps(),
                    pBase,
                    indices,
                    _simd_castsi_ps(mask),
                    4 /* gcc doesn't like sizeof(float) */);
#endif
                pBase += m_attributeStrideInVectors * SIMD_WIDTH;
            }
        }

        return true;
    }

#if ENABLE_AVX512_SIMD16
    bool Assemble_simd16(uint32_t slot, simd16vector verts[])
    {
        SWR_ASSERT(slot < m_numAttributes);

        uint32_t numPrimsToAssemble = PA_TESS::NumPrims();
        if (0 == numPrimsToAssemble)
        {
            return false;
        }

        SIMDSCALARI mask = GenPrimMask(numPrimsToAssemble);

        const float* pBaseAttrib = (const float*)&m_pVertexData[slot * m_attributeStrideInVectors * 4];
        for (uint32_t i = 0; i < m_numVertsPerPrim; ++i)
        {
#if USE_SIMD16_FRONTEND
            SIMDSCALARI indices = _simd16_load_si((const SIMDSCALARI*)m_ppIndices[i]);
#else
            SIMDSCALARI indices = _simd_load_si((const SIMDSCALARI*)m_ppIndices[i]);
#endif

            const float* pBase = pBaseAttrib;
            for (uint32_t c = 0; c < 4; ++c)
            {
#if USE_SIMD16_FRONTEND
                verts[i].v[c] = _simd16_mask_i32gather_ps(
                    _simd16_setzero_ps(),
                    pBase,
                    indices,
                    mask,
                    4 /* gcc doesn't like sizeof(float) */);
#else
                verts[i].v[c].lo = _simd_mask_i32gather_ps(
                    _simd_setzero_ps(),
                    pBase,
                    indices,
                    _simd_castsi_ps(mask),
                    4 /* gcc doesn't like sizeof(float) */);
                verts[i].v[c].hi = _simd_setzero_ps();
#endif
                pBase += m_attributeStrideInVectors * SIMD_WIDTH;
            }
        }

        return true;
    }

#endif
    void AssembleSingle(uint32_t slot, uint32_t primIndex, __m128 verts[])
    {
        SWR_ASSERT(slot < m_numAttributes);
        SWR_ASSERT(primIndex < PA_TESS::NumPrims());

        const float* pVertDataBase = (const float*)&m_pVertexData[slot * m_attributeStrideInVectors * 4];
        for (uint32_t i = 0; i < m_numVertsPerPrim; ++i)
        {
#if USE_SIMD16_FRONTEND
            uint32_t index = useAlternateOffset ? m_ppIndices[i][primIndex + SIMD_WIDTH_DIV2] : m_ppIndices[i][primIndex];
#else
            uint32_t index = m_ppIndices[i][primIndex];
#endif
            const float* pVertData = pVertDataBase;
            float* pVert = (float*)&verts[i];

            for (uint32_t c = 0; c < 4; ++c)
            {
                pVert[c] = pVertData[index];
                pVertData += m_attributeStrideInVectors * SIMD_WIDTH;
            }
        }
    }

    bool NextPrim()
    {
        uint32_t numPrims = PA_TESS::NumPrims();
        m_numPrims -= numPrims;
        m_ppIndices[0] += numPrims;
        m_ppIndices[1] += numPrims;
        m_ppIndices[2] += numPrims;

        return HasWork();
    }

    SIMDVERTEX& GetNextVsOutput()
    {
        SWR_ASSERT(0, "%s", __FUNCTION__);
        static SIMDVERTEX junk;
        return junk;
    }

    bool GetNextStreamOutput()
    {
        SWR_ASSERT(0, "%s", __FUNCTION__);
        return false;
    }

    SIMDMASK& GetNextVsIndices()
    {
        SWR_ASSERT(0, "%s", __FUNCTION__);
        static SIMDMASK junk;
        return junk;
    }

    uint32_t NumPrims()
    {
        return std::min<uint32_t>(m_numPrims, SIMD_WIDTH);
    }

    void Reset() { SWR_ASSERT(0); };

    SIMDSCALARI GetPrimID(uint32_t startID)
    {
#if USE_SIMD16_FRONTEND
        return _simd16_add_epi32(_simd16_set1_epi32(startID), m_vPrimId);
#else
        return _simd_add_epi32(_simd_set1_epi32(startID), m_vPrimId);
#endif
    }

private:
    const SIMDSCALAR*   m_pVertexData = nullptr;
    uint32_t            m_attributeStrideInVectors = 0;
    uint32_t            m_numAttributes = 0;
    uint32_t            m_numPrims = 0;
    uint32_t*           m_ppIndices[3];

    uint32_t            m_numVertsPerPrim = 0;

    SIMDSCALARI         m_vPrimId;
};

// Primitive Assembler factory class, responsible for creating and initializing the correct assembler
// based on state.
template <typename IsIndexedT, typename IsCutIndexEnabledT>
struct PA_FACTORY
{
    PA_FACTORY(DRAW_CONTEXT* pDC, PRIMITIVE_TOPOLOGY in_topo, uint32_t numVerts) : topo(in_topo)
    {
#if KNOB_ENABLE_CUT_AWARE_PA == TRUE
        const API_STATE& state = GetApiState(pDC);
        if ((IsIndexedT::value && IsCutIndexEnabledT::value && (
            topo == TOP_TRIANGLE_STRIP || topo == TOP_POINT_LIST ||
            topo == TOP_LINE_LIST || topo == TOP_LINE_STRIP ||
            topo == TOP_TRIANGLE_LIST)) ||

            // non-indexed draws with adjacency topologies must use cut-aware PA until we add support
            // for them in the optimized PA
            (topo == TOP_LINE_LIST_ADJ || topo == TOP_LISTSTRIP_ADJ || topo == TOP_TRI_LIST_ADJ || topo == TOP_TRI_STRIP_ADJ))
        {
            memset(&indexStore, 0, sizeof(indexStore));
            uint32_t numAttribs = state.feNumAttributes;

            new (&this->paCut) PA_STATE_CUT(pDC, (uint8_t*)&this->vertexStore[0], MAX_NUM_VERTS_PER_PRIM * PA_STATE::SIMD_WIDTH,
                &this->indexStore[0], numVerts, numAttribs, state.topology, false);
            cutPA = true;
        }
        else
#endif
        {
            uint32_t numPrims = GetNumPrims(in_topo, numVerts);
            new (&this->paOpt) PA_STATE_OPT(pDC, numPrims, (uint8_t*)&this->vertexStore[0], MAX_NUM_VERTS_PER_PRIM * PA_STATE::SIMD_WIDTH, false);
            cutPA = false;
        }

    }

    PA_STATE& GetPA()
    {
#if KNOB_ENABLE_CUT_AWARE_PA == TRUE
        if (cutPA)
        {
            return this->paCut;
        }
        else
#endif
        {
            return this->paOpt;
        }
    }

    PA_STATE_OPT paOpt;
    PA_STATE_CUT paCut;
    bool cutPA{ false };

    PRIMITIVE_TOPOLOGY topo{ TOP_UNKNOWN };

    PA_STATE::SIMDVERTEX    vertexStore[MAX_NUM_VERTS_PER_PRIM];
    PA_STATE::SIMDMASK      indexStore[MAX_NUM_VERTS_PER_PRIM];
};
