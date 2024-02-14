mklgemm: main.cpp
	icpx -fsycl -I../../include -DWITHMKL=1  -DMKL_ILP64  -qmkl-ilp64=sequential $< -o $@
