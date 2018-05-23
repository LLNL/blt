/*
 * Foo.hpp
 *
 *  Created on: May 23, 2018
 *      Author: settgast1
 */

#ifndef TESTS_INTERNAL_SRC_COMBINE_STATIC_LIBRARY_TEST_FOO1_HPP_
#define TESTS_INTERNAL_SRC_COMBINE_STATIC_LIBRARY_TEST_FOO1_HPP_

#include<string>

namespace blt_test
{

class Foo1
{
public:
  Foo1();
  ~Foo1();
  
  std::string output();
};

} /* namespace blt_test */

#endif /* TESTS_INTERNAL_SRC_COMBINE_STATIC_LIBRARY_TEST_FOO1_HPP_ */
