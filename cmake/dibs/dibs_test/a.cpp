#include <mpi.h>
#include <caliper/Annotation.h>
int main(){
  cali::Annotation("foo").begin("dog");
  cali::Annotation("foo").end();
}
