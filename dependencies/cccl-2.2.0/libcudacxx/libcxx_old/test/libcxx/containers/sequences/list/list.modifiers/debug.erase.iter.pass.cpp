//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <list>

// Call erase(const_iterator position) with iterator from another container

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-debug-mode, c++03

#include <list>

#include "check_assertion.h"

int main(int, char**) {
    int a1[] = {1, 2, 3};
    std::list<int> l1(a1, a1+3);
    std::list<int> l2(a1, a1+3);
    std::list<int>::const_iterator i = l2.begin();
    TEST_LIBCUDACXX_ASSERT_FAILURE(l1.erase(i), "list::erase(iterator) called with an iterator not referring to this list");

    return 0;
}
