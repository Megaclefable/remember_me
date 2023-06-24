#include <iostream>
#include <vector>

  /* Example : data.size()=4   data[]=[1,4,3,2] , if the i-th value is bigger than (i+1)-th value, change these position.
  i=1 : 
    j=0 -> data[0], data[0+1] treat : data[1,4,3,2] 
  -> j=1 -> data[1], data[1+1] treat : data[1,3,4,2]
  -> j=2 -> data[2], data[2+1] treat : data[1,3,2,4]  (the first j loop is over, data.size=4-1=3, integer j<3)

  i=2 : 
    j=0 -> data[0], data[0+1] treat : data[1,3,2,4]
   -> j=1 -> data[1], data[1+1] treat : data[1,2,3,4]
   -> j=2 -> data[2], data[2+1] treat : data[1,2,3,4] (the second j loop is over, data.size=4-2=3 , integer j<2)

  i=3 : 
    ...

  The biggest element will be at the end of the array. = bubble sort
  O(n^2) -> not efficient , we already know that : if a<b , b<c  then, a<c. 
  But this algorithm is actually  compare a,c for example which is not necessary.
  */
  

using namespace std;

template <typename T>
void sort_list(vector<T>& data) {

/*i index is changing 1 to len(data) -1 */
  for (size_t i = 1; i < data.size(); i++) {
    for (size_t j = 0; j < data.size() - i; j++) {
      if (data[j] > data[j + 1]) {
        // Change the position of the data[j] and data[j + 1].
        T temp = std::move(data[j]);
        data[j] = std::move(data[j + 1]);
        data[j + 1] = std::move(temp);
      }
    }
  }
}

int main() {
  vector<int> s = {1, 9, 8, 5, 4, 6, 7, 3, 2, 10};
  sort_list(s);

  for (int num : s) {
    cout << num << " ";
  }
}

