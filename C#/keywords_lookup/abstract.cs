/*
abstract 한눈에 보기

추상 클래스(abstract class): 직접 인스턴스화 불가. 공통 구현(필드/메서드/속성/생성자 등)을 일부 제공하면서, **반드시 파생 클래스가 구현해야 하는 멤버(추상 멤버)**를 선언할 수 있어요.

추상 멤버(abstract method/property/indexer/event): 본문이 없는 멤버. 반드시 파생 클래스에서 override로 구현해야 합니다. (이미 virtual의 의미를 내포하므로 virtual과 함께 쓰지 않음)

접근 제한: 추상 멤버는 private로 선언할 수 없습니다(파생 클래스가 오버라이드해야 하므로). 보통 protected/public/internal/protected internal/private protected를 사용합니다.

조합 불가: abstract는 멤버 수준에서 static, sealed, virtual, extern과 같이 쓸 수 없습니다.
(클래스 수준에서 abstract sealed는 금지지만, static class는 컴파일러가 내부적으로 abstract sealed로 취급)

abstract override 가능: 중간 단계의 추상 클래스가 **기반의 가상 멤버를 ‘추상 오버라이드’**로 바꿔, 더 아래 단계가 반드시 구현하도록 강제할 수 있어요.

언제 인터페이스 대신 추상 클래스를?

공통 구현을 공유하고 싶을 때(템플릿 메서드, 기본 필드/도우미 메서드 등)

생성자/상태가 필요한 계층 구조일 때

순수 계약만 필요하고 다중 상속 유연성이 더 중요하면 인터페이스가 적합

*/




// Program.cs
using System;
using System.Collections.Generic;

// 1) abstract class, abstract member
abstract class Shape
{
    protected Shape(string name) { Name = name; }

    public string Name { get; }

    // abstract property, child classs should exist.
    public abstract double Area { get; }

    // abstract method, child classs should exist.
    public abstract void Draw();

    // common configuration
    public void Describe()
    {
        Console.WriteLine($"{Name} | Area = {Area:F2}");
    }
}

class Rectangle : Shape
{
    public double Width { get; }
    public double Height { get; }

    public Rectangle(double width, double height) : base("Rectangle")
    {
        Width = width; Height = height;
    }

    public override double Area => Width * Height;

    public override void Draw()
    {
        Console.WriteLine($"Drawing rectangle {Width} x {Height}");
    }
}

class Circle : Shape
{
    public double Radius { get; }

    public Circle(double radius) : base("Circle") => Radius = radius;

    public override double Area => Math.PI * Radius * Radius;

    public override void Draw()
    {
        Console.WriteLine($"Drawing circle r={Radius}");
    }
}

// 2) abstract override memo
class Animal
{
    // Basic configuration is here :
    public virtual void Speak() => Console.WriteLine("(silence)");
}

// It should be configured here.
abstract class Canine : Animal
{
    public abstract override void Speak();
}

// Final detailed class
sealed class Dog : Canine
{
    public override void Speak() => Console.WriteLine("Woof!");
}

class Program
{
    static void Main()
    {
        var shapes = new List<Shape>
        {
            new Rectangle(3, 4),
            new Circle(2.5)
        };

        foreach (var s in shapes)
        {
            s.Describe();
            s.Draw();
        }

        Animal a = new Dog();
        a.Speak();
    }
}




/*
자주하는 실수 :

인스턴스화 금지: new AbstractType()은 컴파일 오류입니다. 항상 구체 하위 타입을 생성하세요.

본문 금지: 추상 멤버에는 메서드 본문/접근자 본문이 없어야 합니다. (세미콜론으로 끝남)

모든 추상 멤버 구현: 비추상 파생 클래스는 상속 계층의 모든 추상 멤버를 override로 구현해야 합니다.

new vs override: 기반 멤버를 바꿔 부르고 싶다면 override를 쓰세요. new는 숨김(hiding)이며 다형적 호출에 참여하지 않습니다.

템플릿 메서드 패턴: 공통 흐름(비추상 메서드)을 정의하고, 세부 단계만 추상 멤버로 강제하면 재사용성과 유연성이 높아집니다.
*/
