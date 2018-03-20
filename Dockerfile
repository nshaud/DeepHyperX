FROM pytorch/pytorch:latest
MAINTAINER Nicolas Audebert (nicolas.audebert@onera.fr)

RUN pip install scipy tqdm seaborn scikit-learn spectral visdom

# Install libGL for matplotlib/seaborn
RUN apt update && apt install -y libgl1-mesa-glx  --no-install-recommends && rm -rf /var/lib/apt/lists/*

#WORKDIR /workspace
#RUN git clone https://gitlab.inria.fr/naudeber/DeepHyperX
WORKDIR /workspace/DeepHyperX/
COPY . .
RUN python main.py --download KSC Botswana PaviaU PaviaC IndianPines

EXPOSE 8097

ENTRYPOINT ["sh", "start.sh"]

