# 📝 C#에서 자주 하는 실수 (Common Mistakes)

이 문서는 C# 초보자가 자주 실수하는 문법과 개념들을 정리한 노트입니다.  
각 항목은 설명 → 코드 예시 순서로 구성되어 있습니다.

---

## ⚡ Fall Through (switch문에서 break 빠뜨림)

**설명:**  
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
