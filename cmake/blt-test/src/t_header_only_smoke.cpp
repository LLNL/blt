
#include "gtest/gtest.h"
#include "HeaderOnly.hpp"

TEST(blt_header_only_smoke,basic_assert_example)
{
    EXPECT_TRUE( blt::ReturnTrue() );
}
