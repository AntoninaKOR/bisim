# docker build -f Dockerfile -t img . && \
# docker run -it --rm -v ~/logdir/docker:/logdir img \
#   python main.py --logdir /logdir/{timestamp} --configs minecraft debug --task minecraft_diamond


# System
RUN apt-get update && apt-get install -y \
  ffmpeg git vim curl software-properties-common grep \
  libglew-dev x11-xserver-utils xvfb wget \
  && apt-get clean

#needs for docker
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore

#python
RUN apt-get update && apt-get install -y python3.11-dev python3.11-venv && apt-get clean
RUN python3.11 -m venv /venv --upgrade-deps
ENV PATH="/venv/bin:$PATH"
RUN pip install -U pip setuptools

# Envs (finger/spin, cheetah/run и walker/walk )
RUN pip install dm_control
RUN pip install git+git://github.com/denisyarats/dmc2gym.git


# Requirements
RUN pip install jax==0.5.0
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Source
RUN mkdir /app
WORKDIR /app
COPY . .
RUN chown -R 1000:root .

#ENTRYPOINT ["sh", "entrypoint.sh"]
