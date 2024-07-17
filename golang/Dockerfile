# Pulling official golang image from hub.docker.com
FROM golang:1.22

WORKDIR /app

# Copying main .mod and .sum files to root dir
COPY go.mod go.sum ./

# Copying server side .mod and .sum files to root dir
COPY server/go.mod server/go.sum ./

# Downloading modules
RUN go mod download

# Copying the remaining files
COPY . .

# Building go executable
RUN go build -v -o go-serving /app/main.go

# Exposing at port :8080
EXPOSE 8080

# CMD executable
CMD ["./go-serving"]