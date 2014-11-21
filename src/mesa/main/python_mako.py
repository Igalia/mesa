try:
    import sys
    import mako
except ImportError as err:
    sys.exit(err)
else:
    ver_min_req_str = str(sys.argv[1]);
    ver_min_req = ver_min_req_str.split('.');
    ver_str = mako.__version__
    ver = ver_str.split('.')

    for i in range(len(ver)):
        if int(ver[i]) < int(ver_min_req[i]):
            sys.exit(1);
        if int(ver[i]) > int(ver_min_req[i]):
            sys.exit(0);
    sys.exit(0);
