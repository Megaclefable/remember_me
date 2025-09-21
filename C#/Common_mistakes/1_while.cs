using System;

class Program
{
    static void Main(string[] args)
    {
        int a = 1;
        for (;;)
        {
            if (a == 1)
            {
                int b = a * 2;  // 
                if (b == 2) break; //  break will be applied to "for loop"
                Console.WriteLine(b);  // this is not applied
            }
            Console.WriteLine("Done1");
        }
        Console.WriteLine("Done2");
    }
}


/* ----------------------------------------------
print result:

Done2
---------------------------------------------- */

// 

using System;

class Program
{
    static void Main(string[] args)
    {
        int a = 1;
        for (;;)
        {
            while(a == 1)
            {
                int b = a * 2;  // 
                if (b == 2) break; //  break will be applied to "while loop"
                Console.WriteLine(b);  //
                break;  //this won't be used.
            }
            Console.WriteLine("Done1");
        }
        Console.WriteLine("Done2");
    }
}

/* ----------------------------------------------
print result:

Done1 and loop
---------------------------------------------- */
