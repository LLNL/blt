#include<iostream>
#include "foo1.hpp"
#include "foo2.hpp"

using namespace blt_test;
int main( int argc, char ** argv )
{
  Foo1 foo1;
  Foo2 foo2;

  std::cout<<foo1.output()<<std::endl;
  std::cout<<foo2.output()<<std::endl;

  return 0;
}
