---

## 1. Fall Through 


**explanation:**  
In C#, if I don't put a break after each case in a switch statement,
it will not automatically fall through to the next case.



```c
int a=0;
switch(0)
{
case 0:
    a++;
case 1:
    a++;
case 2:
    a++;
```

The return value will be 3.
If I run this is c#, there will be an error.


```csharp
int a=0;
switch(0)
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

```

C# works with break.


** `switch` seems ok?

In practice, `switch` statements are rarely used to evaluate objects.  
They are mostly used with numeric values, enums, or strings.  
Since C cannot evaluate strings, switch statements are used only in a limited way.  
However, C# supports string evaluation, which allows `switch` statements to be used for a wider variety of logic.  


```csharp
using System;

class Program
{
    static void convert(ref string color)
    {
        switch(color)
        {
            case "Red" : color = "red"; break;
            case "Blue" : color = :blue"; break;
            default : color = "undefined"; break;
        }
    }

    static void Main(string[] args)
    {
        string s = "Red";
        convert(ref s);
        Console.WriteLine(s);
    }
}
```

-> Avoid `if` and `else if` abuse, too many boolean, so use `switch` instead


```
