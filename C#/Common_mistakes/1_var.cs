
// 1. var

//new keyword, instance


using System;
using System.IO;
using System.Collections.Generic;

class Program
{
  static void Main(string[] args)
  {
    Dictionary<string, Action<TextWriter>> dic = new Dictionary<string, Action<TextWriter>>();
    dic.Add("Sample1", (writer) = {writer.WriteLine("I'm sample1!");});
    dic.Add("Sample2", (writer) = {writer.WriteLine("I'm sample2!");});
    foreach(var item in dic.Values) item(Console.Out);
  }
}



//-> good readability with `var`

using System;
using System.IO;
using System.Collections.Generic;

class Program
{
  static void Main(string[] args)
  {
    var dic = new Dictionary<string, Action<TextWriter>>();
    dic.Add("Sample1", (writer) = {writer.WriteLine("I'm sample1!");});
    dic.Add("Sample2", (writer) = {writer.WriteLine("I'm sample2!");});
    foreach(var item in dic.Values) item(Console.Out);
  }
}


/*
-> 
Using `var` can make code more concise when dealing with complex generic types.
However, overusing `var` when the type is not clear can reduce readability.
*/
