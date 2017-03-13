/*
 * Copyright (c) 2015, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

#include "gtest/gtest.h"
#include "gmock/gmock.h" 

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Interface to Mock
//------------------------------------------------------------------------------
class Thing
{
  public:
      virtual ~Thing() {}
      virtual void Method() = 0;
};

//------------------------------------------------------------------------------
// Interface User
//------------------------------------------------------------------------------
class MethodCaller
{
  public: 
        MethodCaller(Thing *thing)
        :m_thing(thing)
        {
            // empty
        }

        void Go()
        {
            // call Method() on thing 2 times
            m_thing->Method();
            m_thing->Method();
        }

  private:
      Thing* m_thing;
};

//------------------------------------------------------------------------------
// Mocked Interface
//------------------------------------------------------------------------------
class MockThing : public Thing
{
  public:
        MOCK_METHOD0(Method, void());
};


//------------------------------------------------------------------------------
// Actual Test
//------------------------------------------------------------------------------
using ::testing::AtLeast;
TEST(blt_gtest_smoke,basic_mock_test)
{
    MockThing m;
    EXPECT_CALL(m, Method()).Times(AtLeast(2));

    MethodCaller mcaller(&m);

    mcaller.Go();
}


//------------------------------------------------------------------------------
// Main Driver
//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // The following lines must be executed to initialize Google Test
    // and Google Mock before running the tests.
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}
