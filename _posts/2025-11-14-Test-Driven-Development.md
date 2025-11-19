---
title: "TDD: Test-Driven Development"
categories:
  - blog
tags:
  - TDD
  - Test-Driven Development
---

This blog post is a (hopefully short) collection of quick thoughts that come to my mind when I think about TDD. Too often, people are skeptical about it or refrain from using it, but always for the wrong reasons. And it is very easy to tell, just from reading someone's code, if they used TDD or not.

Some of the symptoms:

- Code is hard to import and to use;
- Functions are too long;
- Classes for everything;
- Unreachable code;
- Unhandled use-cases;
- Try/catch everywhere, because you never know...
- And so much more!

Thus, I decided to write this article. Hopefully it will be useful to someone.

## Introduction

There is no robust, bug-free code without good tests.
Tests make sure that any problems are caught before they reach production.

Test-Driven Development (TDD) is an approach where tests are written **before** the code.

What? How? Calm down.

In TDD, a developer writes a failing test for a new feature or change, then writes just enough code to pass that test (and, optionally, refactors the code while test continues to pass).

And the cycle repeats.

In this way, the test _drives_ the code. The test expresses a desire, the code obeys.

Too often, people think about code and tests as two entities that come together. They don't. Tests cause the code. Tests are why the code exist, not the other way around. Understanding this paradigm shift is essential.

In summary, tests are the only reason any source code should exist.
If you are not writing code to pass a test, then you're just creating a bug.

## How does this differ from traditional testing?

Traditional testing tests after coding is done. You code something, then write tests.

This makes the tests (almost) completely useless, because writing tests after coding only ensures that the tests agree with what the code already does. Testing after coding does not challenge any choices that you already made...

What if you write a test after coding, and realize your code does not pass it? In this case, you'd have to change your code, correct. However, your code was born before the test, which means it may be untestable, or hard to change. In this scenario, you would've saved up much more time by writing tests first in the first place. And this scenario is very common.

Writing code after tests also has the following benefit: there is no part of the code the tests are not _listening_ to. Change any part of your code, and a test will know about it. Will it still pass? Ok, but they are always listening. This means you can carelessly edit the code. A test will wake up and fail if you break anything.

In traditional testing, the tests do not _drive_ development, they just validate what already exists.

## Why writing tests before coding

No code should (ever) be written if there is no use-case in mind.
More code means more bugs, so we must write code only when it solves a real problem.
We can risk writing code (writing bugs) when the trade-off is worth it: we write more code, we risk bugs, but we solve something in return.

How can we tell if our code is solving a problem? Only by writing tests **first**.

The development workflow should look something like this:

`Understand logic or requirements` -> `understand how to prove requirements are satisfied` -> `code`.

Thus, it is important that:

- You understand the requirements and are able to write down, in your own natural language, what exactly needs to be done (and why!).
- You understand how to prove the requirements are satisfied. How can you tell that this new feature actually works? If you do not know how to answer, then coding makes no sense. Vague understanding -> code that does nothing.
- You are able to translate your understanding into code: write tests that prove that the new functionality has been successfully implemented.

It is also important your tests cover all possible scenarios that can happen in production, and even more. There must be nothing that can happen in production that is not covered by a test.

Finally, bear in mind that writing tests before coding allows you to desire whatever you want from your code. Do not limit your code to what you think is possible, write any test, write demanding tests, desire anything, then code to make it happen. Test are often seen as _limitations_ or _constraints_ to the code, but they are actually the desires that the code should fulfill.

## TDD: green light, red light

The rules are simple. Start with a test, write just enough test and stop as soon as the test fails. Only when the test fails (red), you can code.
Now, write enough code just to pass the test, not one word more!
If the test passes (green), you cannot write more code. Go back to writing more tests.

Rinse and repeat. Simple.

## What does this look like in practice?

Let's say we want to implement a new feature.

The first thing we need are the requirements:

```text
Build a string manipulator with the following requirements:
- convert to lower case
- remove a defined pattern from the string
```

Without something as clear as this, there is no going forward. If your requirements are too vague or too broad, either re-align with managers and/or colleagues or break it down.

Then, we incorporate this into a test and start writing it:

```python
def test_convert_lower_case() -> None:
    """Test our string manipulator.

    We want to ensure that the following requirements are met:
    - Input `str` is converted to lower case.
    - A defined pattern is removed from the string.

    We will prove that requirements are met by testing the following cases:
    1. If input is not a `str`, raise any error.
    2. Input `str` has upper-case characters; output `str` only has lower-case.
    3. Input `str` has the predefined patter; output `str` does not have it.
    4. Input `str` is already lower-case and does not have the predefined pattern; output `str` is equal to input.
    """
```

You can see there there is no code, just the test set up. You can also see that our ideas are super clear about what feature we want to implement and how we are planning to test that it works.

Now that the plan is clear, we can write the tests.

Yes, before writing the code. This will ensure that the test _drives_ the code: this will prevent us from writing code that is too complex, or code that is hard to use (and test).

### Step 1: decide how we should import your new code

> Remember [this section](#tdd-green-light-red-light).

The first thing we have to do is to decide how the test has to import the function or class to test.
This will force you to make this resource easily accessible and not come up with complex stuff like `from this.that.now_this._private._okhere import _weird_name`.
Imagine if `numpy` or `pandas` forced you to import their code like that... Choose something simple.

```python
from mylib import manipulate_string

def test_convert_lower_case() -> None:
    """ < same - as - above > """
```

Have we written enough test? Let's run this test -> it fails because the resource `manipulate_string` does not exist.

Good. Now we can code.

We create the following:

```python
# Create this file in a place so that it satisfies the import in the test
def manipulate_string() -> str:
    """<describe>"""
```

Stop there. Does the test run and pass? Well, yes, this is enough code to pass the test.
So go back to testing.

### Step 2: write some more test

We can import the function now, let's use it:

```python
from mylib import manipulate_string

def test_convert_lower_case() -> None:
    """ < same - as - above > """
    manipulate_string("PYTEST")
```

If you run this test, it fails because the function `manipulate_string` does not accept any argument yet. So now stop writing the test and code!

Update the code:

```python
def manipulate_string(x: str) -> str:
    """<describe>"""
```

Stop. Run the test, does it pass? Yes. Good. Back to testing.

```python
from mylib import manipulate_string

def test_convert_lower_case() -> None:
    """ < same - as - above > """
    out = manipulate_string("PYTEST")
    assert out == "pytest"
```

The error now will be different: the runtime will complain that the function returned `None` and, thus, is not equal to our target value `"pytest"`.
So we must go back to coding.

Update the code:

```python
def manipulate_string(x: str) -> str:
    """<describe>"""
    return x.lower()
```

This will make the test pass!

So no more coding. Back to the testing.

### Step 3: even more tests

Let's raise an error if input is not a `str`. This was one of the requirements. Are we able to translate this into code? Yes.

```python
import pytest
from mylib import manipulate_string

def test_convert_lower_case() -> None:
    """ < same - as - above > """
    out = manipulate_string("PYTEST")
    assert out == "pytest"

    with pytest.raises(Exception):
        manipulate_string(None)
```

This test won't pass until we do:

```python
def manipulate_string(x: str) -> str:
    """<describe>"""
    if not isinstance(x, str):
        raise Exception("Invalid input.")
    return x.lower()
```

Now it will pass, we go back to testing.

And so on. Until the tests have translated all requirements and cover edge-cases.

## Bad tests, good tests

In the example above, you can see how we are testing behavior, not implementation.

The same test would pass also with this code:

```python
from pydantic import validate_call, ConfigDict

@validate_call(
    config=ConfigDict(validate_assignment=True, validate_default=True),
    validate_return=True,
)
def manipulate_string(x: str) -> str:
    """<describe>"""
    return x.lower()
```

Which solution is better? It does not matter.
The `pydantic`-based solution implicitly brings over more functionalities, but the test is not forbidding them, so both solutions are ok.

If you want to _force_ one solution over another, you must add a test that enforces a functionality that one of the two approaches does not bring. Again, write tests that are driven by a need or a functionality, not to check implementation details.

### Test behavior, not implementation

This is important because we want to test behavior, independent from implementation.
If you tightly couple tests to implementations, classes, etc., then any refactoring is impossible.
Tests will break even if functionalities are the same.

### Test everything against everything

Does this function accept a `str`? Then write tests againt all possible `str` values!
**Especially** meaningless ones.

Good testing starts from testing dummy cases, from testing against non-expected inputs.
Test weirdness first, then go for the actual goal.

Example: this function accepts an `int`... What if I input `None`? What if I input any other type?

Input coverage is very important: test the function against anything that it can accept, and design behavior when something non-expected is inputted.

Let's say the function `call_me()` receives a `str` value, but internally it also reads from a database: then try inputting any `str` value, non-`str` values but also mock the database to return valid but also non-expected data. What do I want this function to do if the databse returns a table with an extra column? What if a column is missing? What if it is empty? What if a `float` column today returns a `str`?

## Code coverage

Code coverage is a metric that tells you how much of your code is tested.
True, it does not tell you how good your tests are, but can be used to spot areas of your code that are untested.
What if these areas contain a typo? What happens if I run them?
This is why you should cover all your code with tests (`100%` coverage).

As said, `100%` coverage only tells you that all your code is tested, not if it is tested well. So `100%` coverage should be the starting point of testing, not the goal.

Many people say: _Why should I cover everything? Coverage is just a metric, if the most important pathways are covered, why should I test everything?_

The answer to this question is another question: if _the most important pathways are covered_, then I could safely remove the uncovered lines from the source code, right? Why have those extra lines?

They never reply. Or they say: but they are being run, in production! At this point you can just reply: have you just admitted you have **untested** code in production?

Tip: if you do TDD, it will be very hard for you to have uncovered code.

What's more, code coverage makes code reviews easier. If your code is `100%` covered, then I do not have to worry for silly things. I can just read your tests and check if they make sense, if they truly prove the new functionalities have been successfully implemented. I may even not read the source code, I will know, just from the tests, that all is well.
