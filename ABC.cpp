#include <iostream>
#include <string>
#include <stdexcept>

using namespace std;

class ABCMeta {
public:
  virtual ~ABCMeta() {}
};

template <typename T>
concept ABCMetaSubclass = requires(T t) {
  { t.register() } -> std::same_as<void>;
  { T::__subclasshook__(T()) } -> std::same_as<bool>;  
};

class Registered: public ABCMeta {
public:
  static bool __subclasshook__(ABCMetaSubclass auto subclass) {
    return subclass.register(); 
  }
};

template<typename T>
class ABC: public Registered {
public:
  static void register() {
    cout << "Registering " << typeid(T).name() << endl;
  }  
};

class MyABC: public ABC<MyABC> {
public:
  void foo() {
    cout << "MyABC::foo" << endl; 
  }
};

int main() {
  
  MyABC myabc;
  myabc.foo();
  
  return 0;
}

class ABCMeta {
public:
  virtual ~ABCMeta() {}
  virtual void foo() = 0;
};

class MyABC: public ABC<MyABC> {
public:
  void foo() override {
    cout << "MyABC::foo" << endl; 
  } 
};

template<typename T>
struct abstractmethod {
  struct registrar {
    registrar(T* t) {
      if(t->foo() != 0) {
        throw logic_error("Method is not abstract!");
      }
    }
  }; 
};

class ABCMeta {
public:
  virtual ~ABCMeta() {}
  virtual int foo() = 0;
  
  abstractmethod<ABCMeta>::registrar reg{this};
};

template<typename T>
concept ABCSubclass = 
  std::derived_from<T, ABCMeta> &&
  requires(T t) {
    t.register();
  };

template<typename T>
class ABC {
public:
  static void register() {
    if constexpr (!ABCSubclass<T>) {
      throw logic_error("Class must be ABC subclass!");
    }
    
    cout << "Registering " << typeid(T).name() << endl;
  }
};

// abc.h
#include <iostream>
#include <exception>
#include <typeinfo>

struct ABCMeta {
  virtual ~ABCMeta() {} 
};

template<typename T>
concept ABCSubclass = 
  std::derived_from<T,ABCMeta> &&
  requires(T t) {
    {t.register()} -> std::same_as<void>;
  };

template<typename T> 
struct abstractmethod {
  struct registrar {
    registrar(T* t) {
      if(!t->is_abstract()) {
        throw std::logic_error("Method must be abstract!");
      }
    }
  };
};

template<typename T>
class ABC {
protected:
  static void register() { 
    if constexpr(!ABCSubclass<T>) {
      throw std::logic_error("Class must be ABC subclass!"); 
    }
    std::cout << "Registering: " << typeid(T).name() << '\n';
  }
public:
  static bool __subclasshook__(ABCSubclass auto subclass) {
    subclass.register();
    return true; 
  }
};

// concrete_abc.h
#include "abc.h"

class MyABC : public ABC<MyABC> { 
public:
  virtual bool is_abstract() { return false; }
};

int main() {
  
  MyABC c;
  
  if(ABC<MyABC>::__subclasshook__(c)) {
    std::cout << "Is ABC subclass!\n";
  }
  
  return 0;
}

