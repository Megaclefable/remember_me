#include <vector>
#include <iostream>

/* appends all the elements of A starting from index 1 to the end of the sorted vector. */

/*An iterator is similar to a pointer and it's used to point to elements in a container like a vector, list, set, etc.
The expression A.begin() returns an iterator pointing to the first element of the vector A. 

When you add an integer current_a to this iterator, it moves the iterator current_a positions forward in the vector. 
So A.begin() + current_a is an iterator pointing to the current_ath element of A.

For example, if current_a is 2, A.begin() + current_a points to the third element of A (as counting starts from 0). 
This iterator is then used as the start position for the range of elements to be inserted into another vector. */

int main() {
    std::vector<int> A = {3, 4, 5};
    std::vector<int> B = {1, 2};
    int current_a = 1;

    std::vector<int> sorted = B;
    sorted.insert(sorted.end(), A.begin() + current_a, A.end());

    for (int i : sorted) {
        std::cout << i << " ";  // prints: 1 2 4 5 
    }
    return 0;
}
