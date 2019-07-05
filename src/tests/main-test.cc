#include <gtest/gtest.h>
#include "tests/params-for-test.h"

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  emt6ro::TEST_DATA_DIR = std::getenv("TEST_DATA_DIR");
  if (!emt6ro::TEST_DATA_DIR) {
    std::cerr << "TEST_DATA_DIR not specified." << std::endl;
    exit(1);
  }
  return RUN_ALL_TESTS();
}
