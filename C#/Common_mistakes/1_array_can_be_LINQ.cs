
/*Instead of using loop for array, we can use LINQ */

/* Array with loop */
using System;

class Program
{
  static void Main(string[] args)
  {
    int[] array= {1,-1,2,-2,3};
    foreach(var item in array)
    {
      if(item<0)
      {
        Console.WirteLine(item);
        break;
      }
    }
  }
}

/* Array with LINQ, first data which satisfies the given condition */
using System;
using System.Linq;

class Program
{
  static void Main(string[] args)
  {
    int[] array= {1,-1,2,-2,3};
    Console.WiteLine(array.FirstOrDefault(c => c<0));
  }
}


/* Array with LINQ, the data which satisfies the given condition at n_th position */
using System;
using System.Linq;

class Program
{
  static void Main(string[] args)
  {
    int[] array= {1,-1,2,-2,3};
    Console.WiteLine(array.Where(c=>c<0).ElementAt(1));
  }
}

//Output is -2
// LastOrDefault -> last value
