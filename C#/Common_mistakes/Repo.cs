
using System;
using System.Collections.Generic;
using System.Linq;

namespace RepoDemo
{
    // Common entity contract (all entities must have an Id)
    public interface IEntity
    {
        int Id { get; set; }
    }

    // Base class containing common properties/behavior
    public abstract class BaseEntity : IEntity
    {
        public int Id { get; set; }
        public DateTime CreatedAt { get; private set; } = DateTime.UtcNow;

        public override string ToString() => $"{GetType().Name}#{Id} (CreatedAt UTC: {CreatedAt:O})";
    }

    // Customer entity
    public class Customer : BaseEntity
    {
        public string Name { get; set; } = "";
        public override string ToString() => base.ToString() + $", Name={Name}";
    }

    // Order entity
    public class Order : BaseEntity
    {
        public int CustomerId { get; set; }
        public decimal Amount { get; set; }
        public override string ToString() => base.ToString() + $", CustomerId={CustomerId}, Amount={Amount}";
    }

    // Repository interface (defines CRUD contract)
    public interface IRepository<T> where T : BaseEntity, new()
    {
        /// <summary>
        /// Adds a new entity to the repository and returns the stored entity.
        /// </summary>
        T Add(T entity);

        /// <summary>
        /// Retrieves an entity by its ID. Returns null if not found.
        /// </summary>
        T? Get(int id);

        /// <summary>
        /// Returns all entities, or optionally only those matching the given predicate.
        /// </summary>
        IEnumerable<T> GetAll(Func<T, bool>? predicate = null);

        /// <summary>
        /// Deletes the entity with the specified ID.
        /// </summary>
        void Remove(int id);

        /// <summary>
        /// Updates an existing entity. Throws an exception if it does not exist.
        /// </summary>
        T Update(T entity);
    }

    // In-memory generic Repository implementation
    public class InMemoryRepository<T> : IRepository<T> where T : BaseEntity, new()
    {
        private readonly Dictionary<int, T> _store = new();
        private int _nextId = 1;

        /// <summary>
        /// Adds a new entity and automatically assigns an ID.
        /// </summary>
        public T Add(T entity)
        {
            entity.Id = _nextId++;
            _store[entity.Id] = entity;
            return entity;
        }

        /// <summary>
        /// Returns the entity with the given ID, or null if it doesn’t exist.
        /// </summary>
        public T? Get(int id) => _store.TryGetValue(id, out var v) ? v : null;

        /// <summary>
        /// Returns all entities, or filtered results if a predicate is provided.
        /// </summary>
        public IEnumerable<T> GetAll(Func<T, bool>? predicate = null)
            => predicate is null ? _store.Values.ToList() : _store.Values.Where(predicate).ToList();

        /// <summary>
        /// Removes the entity with the given ID from the repository.
        /// </summary>
        public void Remove(int id) => _store.Remove(id);

        /// <summary>
        /// Updates an existing entity. Throws InvalidOperationException if the ID doesn’t exist.
        /// </summary>
        public T Update(T entity)
        {
            if (!_store.ContainsKey(entity.Id))
                throw new InvalidOperationException($"Entity id={entity.Id} not found");

            _store[entity.Id] = entity;
            return entity;
        }
    }

    class Program
    {
        /// <summary>
        /// Application entry point (Main method).  
        /// Demonstrates CRUD operations on customer and order data.
        /// </summary>
        static void Main()
        {
            IRepository<Customer> customers = new InMemoryRepository<Customer>();
            IRepository<Order> orders = new InMemoryRepository<Order>();

            // Add customers
            var alice = customers.Add(new Customer { Name = "Alice" });
            var bob   = customers.Add(new Customer { Name = "Bob" });
            Console.WriteLine($"Added: {alice}");
            Console.WriteLine($"Added: {bob}");

            // Add orders
            orders.Add(new Order { CustomerId = alice.Id, Amount = 120.50m });
            orders.Add(new Order { CustomerId = alice.Id, Amount = 80.00m });
            orders.Add(new Order { CustomerId = bob.Id,   Amount = 42.99m });

            // Query (filtering)
            var aliceOrders = orders.GetAll(o => o.CustomerId == alice.Id);
            Console.WriteLine("\nOrders for Alice:");
            foreach (var o in aliceOrders)
                Console.WriteLine($" - {o}");

            // Update
            bob.Name = "Bobby";
            customers.Update(bob);
            Console.WriteLine($"\nUpdated: {customers.Get(bob.Id)}");

            // Delete
            customers.Remove(alice.Id);
            Console.WriteLine($"\nAfter removing Alice, all customers:");
            foreach (var c in customers.GetAll())
                Console.WriteLine($" - {c}");

            // Thanks to type constraints, invalid types are blocked at compile time.
        }
    }
}
```

---

Would you like me to make the English comments *more concise and natural for native developers* (e.g., as if it were a production-ready open-source project)?
