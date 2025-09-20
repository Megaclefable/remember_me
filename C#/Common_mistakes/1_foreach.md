---

## 1. foreach

**Explanation:**  


With `for` :
```csharp
using System;

class Program
{
  static void Main(string[] args)
  {
    int[] a = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    for(int i=0; i<a.Length; i++) Console.WriteLine(a[i]);
  }
}

```

With `foreach` :

```csharp
using System;

class Program
{
  static void Main(string[] args)
  {
    int[] a = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    foreach(var item in a) Console.WriteLine(item);
  }
}
```

`for` loop: Repeats until a certain condition is satisfied.  
`foreach` loop: Retrieves each element one by one from an enumerable interface.

if the loop is based on the index, the data should be read upto the end of the collection.

`foreach` case : Sequential access  
Step1 : give me the next data  
Step2 : end?  
Step3 : then we are going back to Step1  

So, there is no "next data".

---


`for` can be used as `foreach`  

`for` is used : 

```csharp
using System;
using System.Linq;
using System.Threading.Tasks;

class Program
{
  static void Main(string[] args)
  {
    var heavyQuery = Enumerable.Range(0, 10).Where(c=>
    {
      Task.Delay(1000).Wait(); // Let's say that heavy work is going on
      return true;
    });
    var start= DateTime.Now;
    var enumerator = heavyQuery.GetEnumerator();
    for (; enumerator.MoveNext();)
    {
        Console.Wirte(enumerator.Current);
    }
    Console.WriteLine("time gone : {0}", DateTime.Now - start);
  }
}
```


`foreach` is used : 

```csharp
using System;
using System.Linq;
using System.Threading.Tasks;

class Program
{
  static void Main(string[] args)
  {
    var heavyQuery = Enumerable.Range(0, 10).Where(c=>
    {
      Task.Delay(1000).Wait(); // Let's say that heavy work is going on
      return true;
    });
    var start= DateTime.Now;
    foreach(var item in heavyQuery)
    {
        Console.Write(item);
    }
    Console.WriteLine("time gone : {0}", DateTime.Now - start);
  }
}
```

