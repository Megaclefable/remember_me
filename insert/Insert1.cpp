/* Output : 
  Merged vector:
  4 2 9 8 7 
*/

#include <iostream>
#include <vector>



int main() {
    // Create two vectors with integer elements
    std::vector<int> A = {4, 2, 9};
    std::vector<int> B = {1, 5, 8, 7};

    // Create an empty vector to store the merged result
    std::vector<int> merged;

    // Copy all elements from A into merged
    merged.insert(merged.end(), A.begin(), A.end());

    // Initialize current_b
    int current_b = 2; // Let's suppose we only want to copy from the third element of B

    // Insert the rest of B into merged, starting from current_b
    merged.insert(merged.end(), B.begin() + current_b, B.end());

    // Display the merged vector
    std::cout << "Merged vector:\n";
    for(int i : merged) {
        std::cout << i << ' ';
    }
    std::cout << '\n';

    return 0;
}
