# RPLEX

Rust bindings for
[CPLEX](https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/),
an IP/MIP/QP solver from IBM. Currently highly experimental but
working for IP/MIP problems.

```toml
[dependencies]
rplex = "0.1" # not actually on crates.io yet
```

You *must* have CPLEX installed for RPLEX to work. The build script
attempts to locate CPLEX automatically (and should do so assuming
you've installed it at the usual location on Linux). CPLEX is not
necessary to run a binary compiled with RPLEX; static linking is used.

*Documentation under construction*

# Example

```rust
fn lpex1() {
    // create a CPLEX environment
    let env = Env::new().unwrap();
    // populate it with a problem
    let mut prob = Problem::new(&env, "lpex1").unwrap();
    // maximize the objective
    prob.set_objective_type(ObjectiveType::Maximize).unwrap();
    // create our variables
    let x1 = prob.add_variable(var!(0.0 <= "x1" <= 40.0 -> 1.0)).unwrap();
    let x2 = prob.add_variable(var!("x2" -> 2.0)).unwrap();
    let x3 = prob.add_variable(var!("x3" -> 3.0)).unwrap();
    println!("{} {} {}", x1, x2, x3);

    // add our constraints
    prob.add_constraint(con!("c1": 20.0 >= (-1.0) x1 + 1.0 x2 + 1.0 x3)).unwrap();
    prob.add_constraint(con!("c2": 30.0 >= 1.0 x1 + (-3.0) x2 + 1.0 x3)).unwrap();

    // solve the problem
    let sol = prob.solve().unwrap();
    println!("{:?}", sol);
    // values taken from the output of `lpex1.c`
    assert!(sol.objective == 202.5);
    assert!(sol.variables == vec![VariableValue::Continuous(40.0),
                                    VariableValue::Continuous(17.5),
                                    VariableValue::Continuous(42.5)]);
}
```

(Translated from the CPLEX example code `lpex1.c`. Taken from an
[actual test](src/lib.rs#L704-L724))

# License

IBM, CPLEX and associated items are (c) IBM.

Copyright (c) 2016, J. David Smith All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
