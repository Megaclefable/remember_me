/* async with no thread */

/* 9 to 0 count down function,
Each count down 1 sec wait, 
parallel count down without using threads*/


// Case 1 :
using System; 
using System.Threading; 
using System.Threading.Tasks;

class Program
{
  class countDownWrapper
  {
    public AutoResetEvent Done = new AutoResetEvent(false);  // work done is initialized as false
    private int count = 9;
    
    public void CountDown()
    {
      Console.WriteLine(count--);
      if(count>=0)
        Task.Delay(1000).ContinueWith((c) =>
        {
          CountDown();
        });
      else Done.Set();
    }
  }

  static void Main(string[] args)
  {
    var a = new countDownWrapper();
    var b = new countDownWrapper();
    a.CountDown();
    b.CountDown();
    WaitHandle.WaitAll(new WaitHandle[]{ a.Done, b.Done }); //Cancel logic should be always configured.
    //AutoResetEvent.WaitAll(new[] {a.Done, b.Done});
  }
}

//There will be an issue with continueWith when it comes to the exception -> unobserved in ContinueWith
// WinForms case, no UI context restore : callback is done in threadpool
//

// Case 2 :
using System; 
using System.Threading;
class Program
{
  //async used.
  private static async Task countDown()
  {
    for(int i = 9; i>=0; i--)
    {
      Console.WriteLine(i);
      await Task.Delay(1000);  //not taking the thread
    } //after loop, the threads are completed. 
  }

  static void Main(string[] args)
  {
    var a = countDown();
    var b = countDown();
    //blocking way
    Task.WaitAll(a,b); // =~ await Task.WhenAll(a, b);
  }
}

// async, await can be used with for.
// async, await -> no need to wait, it returns
