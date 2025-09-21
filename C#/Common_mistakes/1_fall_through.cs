// 1_FallThrough.cs

using System;

namespace FallThroughStudy
{
    class Program
    {
        static void Main(string[] args)
        {
            Demo1_CFallThrough();
            Console.WriteLine(new string('-', 60));
            Demo2_CSharpSwitchWithBreak();
            Console.WriteLine(new string('-', 60));
            Demo3_SwitchWithStrings();
        }

        /* --------------------------------------------------------
         * 1) C-style fall-through behavior (conceptual)
         * --------------------------------------------------------
         * In C (not C#), a switch without 'break' automatically falls
         * through to the next case.
         * 
         * Example (C code):
         * int a = 0;
         * switch(0) {
         *   case 0: a++;   // falls through
         *   case 1: a++;   // falls through
         *   case 2: a++;   // end
         * }
         * Result: a == 3
         * 
         * In C#, this would be a compile-time error unless each case
         * explicitly ends (with break, goto, return, or throw).
         * -------------------------------------------------------- */
        static void Demo1_CFallThrough()
        {
            Console.WriteLine("Demo1: C-style fall through (illustration only)");
            Console.WriteLine(" -> In C, result would be 3.");
            Console.WriteLine(" -> In C#, this code does not compile without 'break'.");
        }

        /* --------------------------------------------------------
         * 2) C# requires 'break' (or other exit) in each case
         * --------------------------------------------------------
         * Below, C# code compiles because every case ends with break.
         * -------------------------------------------------------- */
        static void Demo2_CSharpSwitchWithBreak()
        {
            Console.WriteLine("Demo2: C# switch requires explicit breaks");

            int a = 0;
            switch (0)
            {
                case 0:
                    a++;
                    break;
                case 1:
                    a++;
                    break;
                case 2:
                    a++;
                    break;
            }
            Console.WriteLine($"Result a = {a}");
        }

        /* --------------------------------------------------------
         * 3) C# switch supports strings (unlike C)
         * --------------------------------------------------------
         * Switch statements in C# can evaluate strings, enums, etc.
         * Example: mapping string values.
         * -------------------------------------------------------- */
        static void Demo3_SwitchWithStrings()
        {
            Console.WriteLine("Demo3: Switch on string values");

            string s = "Red";
            ConvertColor(ref s);
            Console.WriteLine($"Converted string = {s}");

            s = "Blue";
            ConvertColor(ref s);
            Console.WriteLine($"Converted string = {s}");

            s = "Green";
            ConvertColor(ref s);
            Console.WriteLine($"Converted string = {s}");
        }

        // A simple method demonstrating switch with string
        static void ConvertColor(ref string color)
        {
            switch (color)
            {
                case "Red":
                    color = "red";
                    break;
                case "Blue":
                    color = "blue"; // corrected typo
                    break;
                default:
                    color = "undefined";
                    break;
            }
        }
    }
}

/* ------------------------------------------------------------
Notes (from 1_FallThrough.md, adapted as comments here):

- In C, switch statements fall through by default if you don't
  add 'break'. This can accumulate case executions.

- In C#, switch statements REQUIRE an exit (break, goto, return,
  or throw). Fall-through is not allowed except when explicitly
  using 'goto case ...'.

- C switch: works only with numeric/integral types, enums.
- C# switch: works with numeric values, enums, and strings too.

- This allows C# to use switch in a wider variety of logic cases,
  avoiding excessive 'if / else if' chains.
------------------------------------------------------------ */
