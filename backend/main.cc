#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include <iostream>
#include <memory>
#include <string>

#include "server.h"

int main(int /*argc*/, char*[] /*argv*/) {
  std::string server_address("0.0.0.0:50051");

  weft::CudaDriverImpl service;

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();

  return EXIT_SUCCESS;
}
