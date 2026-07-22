#pragma once

// Node 2 StandardScaler parameters (4 features)
// Features: bus_V, current_mA_abs, power_mW, lux

const int NODE2_NUM_FEATURES = 4;

const float NODE2_FEATURE_MEAN[] = {
  3.29264290f,
  4.93928576f,
  16.00000000f,
  54.61303602f,
};

const float NODE2_FEATURE_STD[] = {
  0.00165213f,
  0.11600268f,
  1.00000000f,
  113.01523662f,
};
