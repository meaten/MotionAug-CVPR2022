// Util.h
#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

namespace Peaks {
    const float EPS = 2.2204e-16f;
    void findPeaks(vector<float> x0, vector<int>& peakInds);
}

#endif
