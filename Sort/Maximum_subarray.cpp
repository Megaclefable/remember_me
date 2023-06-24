// the maximum should pass the mid point
template <typename T>
T max_subarray_cross_mid(const vector<T>& data, size_t mid) {
  T left_max = data[mid - 1];
  T current_sum = data[mid - 1];

  // i type should be int , not unsigned_t because in case <0.
  int i = mid - 2;
  while (i >= 0) {
    current_sum += data[i--];
    if (current_sum > left_max) {
      left_max = current_sum;
    }
  }

  T right_max = data[mid];
  current_sum = data[mid];
  i = mid + 1;
  while (i < data.size()) {
    current_sum += data[i++];
    if (current_sum > right_max) {
      right_max = current_sum;
    }
  }
  return left_max + right_max;
}

/*--------------------------------------------------------------------------------------------------------*/
template <typename T>
T max_subarray(const vector<T>& data) {
  if (data.size() == 1) {
    if (data[0] > 0) {
      return data[0];
    } else {
      return 0;
    }
  } else if (data.size() == 0) {
    return 0;
  }

  size_t mid = data.size() / 2;
  T left_max = max_subarray(vector<T>(data.begin(), data.begin() + mid));
  T right_max = max_subarray(vector<T>(data.begin() + mid, data.end()));
  T mid_max = max_subarray_cross_mid(data, mid);

  return max(max(left_max, right_max), mid_max);
}

int main() {
  cout << max_subarray(vector<int>{3, -4, 5, -2, -2, 8, -5, -1, 4});
}
