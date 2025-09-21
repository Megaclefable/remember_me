// 1_foreach.cs

using System;
using System.Linq;
using System.Threading.Tasks;

namespace ForeachStudy
{
    class Program
    {
        static void Main(string[] args)
        {
            // Run all demos in order. You can comment out any you don't want.
            Demo1_ArrayLoops();
            Console.WriteLine();
            Console.WriteLine(new string('-', 60));
            Demo2_ForWithEnumerator();
            Console.WriteLine(new string('-', 60));
            Demo3_ForeachDeferredQuery();
        }

        /* --------------------------------------------------------
         * 1) for vs foreach (basic array iteration)
         * --------------------------------------------------------
         * - for   : index-based iteration; runs until the loop condition fails.
         * - foreach: pulls ("yields") the next element from an enumerable.
         *   Think: "Give me next data → Is it the end? → If not, repeat."
         *   There is no explicit index unless you introduce one yourself.
         * -------------------------------------------------------- */
        static void Demo1_ArrayLoops()
        {
            Console.WriteLine("Demo1: for vs foreach with a simple int[]");

            int[] a = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            // With 'for' (index-based access)
            Console.Write("[for]     ");
            for (int i = 0; i < a.Length; i++)
            {
                Console.Write(a[i]);
                if (i < a.Length - 1) Console.Write(", ");
            }
            Console.WriteLine();

            // With 'foreach' (sequential access)
            Console.Write("[foreach] ");
            foreach (var item in a)
            {
                Console.Write(item);
                if (item != a[^1]) Console.Write(", ");
            }
            Console.WriteLine();
        }

        /* --------------------------------------------------------
         * 2) Using 'for' like 'foreach' via IEnumerator
         * --------------------------------------------------------
         * - You can simulate foreach by manually asking the enumerator
         *   to MoveNext() and then reading Current.
         * - This shows how foreach works under the hood for enumerables.
         * -------------------------------------------------------- */
        static void Demo2_ForWithEnumerator()
        {
            Console.WriteLine("Demo2: Simulate foreach using for + IEnumerator");

            var heavyQuery = Enumerable.Range(0, 10).Where(c =>
            {
                // Pretend a heavy operation is happening
                Task.Delay(100).Wait(); // shorter delay to keep sample quick
                return true;            // keep all items
            });

            var start = DateTime.Now;
            var enumerator = heavyQuery.GetEnumerator();

            Console.Write("[for + IEnumerator] ");
            for (; enumerator.MoveNext(); )
            {
                Console.Write(enumerator.Current);
                if (enumerator.Current != 9) Console.Write(", ");
            }
            Console.WriteLine();

            Console.WriteLine($"time gone : {DateTime.Now - start}");
        }

        /* --------------------------------------------------------
         * 3) foreach with deferred execution (LINQ)
         * --------------------------------------------------------
         * - LINQ queries (like Where) are typically deferred: the work
         *   happens when you iterate (MoveNext) — not when you build them.
         * - foreach naturally drives that iteration.
         * -------------------------------------------------------- */
        static void Demo3_ForeachDeferredQuery()
        {
            Console.WriteLine("Demo3: Foreach consuming a deferred LINQ query");

            var heavyQuery = Enumerable.Range(0, 10).Where(c =>
            {
                Task.Delay(100).Wait(); // simulate heavy work per element
                return true;
            });

            var start = DateTime.Now;

            Console.Write("[foreach] ");
            foreach (var item in heavyQuery)
            {
                Console.Write(item);
                if (item != 9) Console.Write(", ");
            }
            Console.WriteLine();

            Console.WriteLine($"time gone : {DateTime.Now - start}");
        }
    }
}

/* ------------------------------------------------------------
Notes (from 1_foreach.md, adapted as comments here):

- for loop: Repeats until a condition becomes false.
- foreach loop: Pulls one element at a time from an enumerable source.
  Sequential access: "Give me next data → Is it the end? → If not, repeat."

- You can emulate foreach using an enumerator:
    var enumerator = sequence.GetEnumerator();
    for (; enumerator.MoveNext(); )
        Use(enumerator.Current);

- LINQ queries are typically deferred; work happens upon enumeration.
------------------------------------------------------------ */
