FROM ubuntu:16.04

# Setup pyenv
ENV PYENV_ROOT /root/.pyenv
ENV PATH /root/.pyenv/shims:/root/.pyenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN apt-get update && \
    apt-get install -y git mercurial build-essential libssl-dev libbz2-dev libreadline-dev libsqlite3-dev curl && \
    curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
RUN pyenv install 3.5.2
RUN pyenv global 3.5.2

# Setup gsutil - used for downloading and updating data
RUN CLOUD_SDK_REPO="cloud-sdk-$(. /etc/os-release && echo $VERSION_CODENAME)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y google-cloud-sdk

# Clean up
RUN apt-get clean \
    && (rm -f /var/cache/apt/archives/*.deb \
        /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin || true)

WORKDIR /opt/mlab

COPY . .

RUN apt-get install -y libffi-dev

RUN pip3 install -r ./pip-requirements.txt && rm -rf /root/.cache/pip /tmp/pip*

CMD ["./start.sh", "9000"]
