using System;

class Program
{
  private static void sample(int a)
  {
    while(a>=0)
    {
      Console.Write(a--);
    }
  }

  static void Main(string[] args)
  {
    sample(9);
    sample(-1);
  }
}

//If I am looking for the sample(-1); at least once,  do is good.


using system; 

class Program
{
  private static void sample(int a)
  {
    do
    {
      Console.WriteLine(a--);
    }
    while(a>=0);
  }
  static void Main(string[] args)
  {
    sample(9);
    sample(-1);
  }
}

// do -> once execution -> while -> back to the sentence in do.

// bool first = true; while (a>=0 || first)... <- not necessary
