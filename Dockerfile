FROM node:19

ARG USERNAME=app
ARG USER_UID=1001
ARG USER_GID=1001

WORKDIR /app
COPY ./ ./
RUN rm ./src/config/config.js && mv ./src/config/config_docker.js ./src/config/config.js
RUN npm install
RUN npm run build

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME 
RUN chown -R $USER_UID:$USER_GID /app

EXPOSE 44444
USER app
CMD ["npm","run","start"]
