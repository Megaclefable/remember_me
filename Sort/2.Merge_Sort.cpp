#include <iostream>
#include <vector>

/* push_back : append, one by one at the end */

/* example : A[]= [1,4,3,8], B[] = [2,7,5,9], merged[]= [] : merged.size()=8
  current_a = 0, current_b = 0 :
  -> A[0]<=B[0]? -> yes -> push (0)_th element in A[] into merged[] : merged[1] -> current_a++
  -> A[1]<=B[0]? ->  no -> push (0)_th element in B[] into merged[] : merged[1,2] -> current_b++
  -> A[1]<=B[1]? -> yes -> push (1)_th element in A[] into merged[] : merged[1,2,4] -> current_a++
  -> A[2]<=B[1]? -> yes -> push (2)_th element in A[] into merged[] : merged[1,2,4,3] -> current_a++
  -> A[3]<=B[1]? -> no  -> push (1)_th element in B[] into merged[] : merged[1,2,4,3,7] -> current_b++
  -> A[3]<=B[2]? -> no  -> push (2)_th element in B[] into merged[] : merged[1,2,4,3,7,5] -> current_b++
  -> A[3]<=B[3]? -> yes -> push (1)_th element in A[] into merged[] : merged[1,2,4,3,7,5,8] -> current_a++
  -> current_a become 4 which is the same with A.size() -> put the rest of the element(s) in B[] from the current_b at the end of merged.
     : merged[1,2,4,3,7,5,8,9]

  *** O(n), But if the array which I want to sort -> divide by 2 : A[] and B[] and apply it separately.
  
*/
using namespace std;

// This returns the vector which is merged.
template <typename T>
vector<T> merge(const vector<T>& A, const vector<T>& B) {
  if (A.empty()) {
    return B;
  } else if (B.empty()) {
    return A;
  }

  vector<T> merged;
  merged.reserve(A.size() + B.size());

  size_t current_a = 0, current_b = 0;
  while (current_a < A.size() && current_b < B.size()) {
    
    if (A[current_a] <= B[current_b]) {
        /*current_a++ : start from inserting 0, not 1 */ 
      merged.push_back(A[current_a++]);
      if (current_a == A.size()) {
        merged.insert(merged.end(), B.begin() + current_b, B.end());
        return merged;
      }
    } 
    
    else if (A[current_a] > B[current_b]) {
      merged.push_back(B[current_b++]);
      if (current_b == B.size()) {
        merged.insert(merged.end(), A.begin() + current_a, A.end());
        return merged;
      }
    }
  }
  return merged;
}

int main() {
  vector<int> A = {2, 5, 7, 10};
  vector<int> B = {1, 3, 8, 9};
  vector<int> merged = merge(A, B);

  for (int num : merged) {
    cout << num << " ";
  }
}
