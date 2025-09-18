---

## 1. Fall Through 


**explanation:**  
C#의 `switch` 문은 각 `case` 뒤에 `break`를 넣지 않으면  
자동으로 다음 `case`로 넘어가지 않습니다. (`fall through`가 안 됨)



```csharp
//  잘못된 예시: break가 없어서 컴파일 에러 발생
switch (x)
{
    case 1:
        Console.WriteLine("One");
    case 2:
        Console.WriteLine("Two");
        break;
}

// 올바른 예시
switch (x)
{
    case 1:
        Console.WriteLine("One");
        break;
    case 2:
        Console.WriteLine("Two");
        break;
}
