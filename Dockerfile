# Use the PorePy development image as the base.
FROM porepy/dev:latest

# Install additional dependencies.
RUN pip install --no-cache-dir pyamg

# Set the working directory inside the container.
WORKDIR /workdir/porepy

# Clone and checkout the CF Verification Experiment branch from PorePy.
RUN git remote add cf_repo https://github.com/pmgbergen/porepy.git
RUN git fetch cf_repo 
RUN git switch -c CF-verification-experiment cf_repo/CF-verification-experiment

# Set up the runscript repository.
WORKDIR /workdir/
RUN git clone https://github.com/mikeljordan/UGS_runscript.git

# Set the working directory to the UGS runscript repo.
WORKDIR /workdir/UGS_runscript

# Set an entrypoint to the directory with the runscript.
ENTRYPOINT ["bash"]
