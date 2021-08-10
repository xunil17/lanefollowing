FROM lgsvl/lanefollowing:latest

RUN pip install numba

WORKDIR /openpilot
RUN git clone https://github.com/commaai/openpilot -b v0.6.6 .

RUN curl http://repo.ros2.org/repos.key | sudo apt-key add -
RUN apt-get update && apt install -y --no-install-recommends \
    beignet-opencl-icd \
    ocl-icd-libopencl1 \
    clinfo \
    clang

# WORKDIR /openpilot/phonelibs
# RUN sudo sh install install_capnp.sh

WORKDIR /openpilot/selfdrive/controls/lib/lateral_mpc
RUN find ./ -name "*.o" | xargs rm
RUN make all

ENV PYTHONPATH $PYTHONPATH:/openpilot/

RUN pip install -U \
    Cython==0.29.15 \
    pycapnp==0.6.4 

WORKDIR /lanefollowing