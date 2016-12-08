extern crate libc;
use libc::{c_int, c_char, c_double};
use std::ffi::CString;

pub const INFINITY: f64 = 1.0E+20;

enum CEnv {}
enum CProblem {}

// #[link(name="cplex", kind="static")]
#[allow(non_snake_case)]
#[allow(dead_code)]
extern "C" {
    // instantiation
    fn CPXopenCPLEX(status: *mut c_int) -> *mut CEnv;
    fn CPXcreateprob(env: *mut CEnv, status: *mut c_int, name: *const c_char) -> *mut CProblem;
    fn CPXsetintparam(env: *mut CEnv, param: c_int, value: c_int) -> c_int;
    // adding variables and constraints
    fn CPXnewcols(env: *mut CEnv, lp: *mut CProblem, count: c_int,
                   obj: *const c_double, lb: *const c_double, ub: *const c_double,
                   xctype: *const c_char, name: *const *const c_char) -> c_int;
    fn CPXaddrows(env: *mut CEnv, lp: *mut CProblem,
                   col_count: c_int, row_count: c_int, nz_count: c_int,
                   rhs: *const c_double, sense: *const c_char,
                   rmatbeg: *const c_int, rmatind: *const c_int, rmatval: *const c_double,
                   col_name: *const *const c_char, row_name: *const *const c_char) -> c_int;
    // setting objective
    fn CPXchgobj(env: *mut CEnv, lp: *mut CProblem, cnt: c_int, indices: *const c_int, values: *const c_double) -> c_int;
    fn CPXchgobjsen(env: *mut CEnv, lp: *mut CProblem, maxormin: c_int) -> c_int;
    // solving
    fn CPXlpopt(env: *mut CEnv, lp: *mut CProblem) -> c_int;
    fn CPXmipopt(env: *mut CEnv, lp: *mut CProblem) -> c_int;
    // getting solution
    fn CPXgetstat(env: *mut CEnv, lp: *mut CProblem) -> c_int;
    fn CPXgetobjval(env: *mut CEnv, lp: *mut CProblem, objval: *mut c_double) -> c_int;
    fn CPXgetx(env: *mut CEnv, lp: *mut CProblem, x: *mut c_double, begin: c_int, end: c_int) -> c_int;
    fn CPXsolution(env: *mut CEnv, lp: *mut CProblem, lpstat_p: *mut c_int, objval_p: *mut c_double,
                   x: *mut c_double, pi: *mut c_double, slack: *mut c_double, dj: *mut c_double) -> c_int;
    // debugging
    fn CPXgeterrorstring(env: *mut CEnv, errcode: c_int, buff: *mut c_char) -> *mut c_char;
    fn CPXwriteprob(env: *mut CEnv, lp: *mut CProblem, fname: *const c_char, ftype: *const c_char) -> c_int;
}

fn errstr(env: *mut CEnv, errcode: c_int) -> Option<String> {
    unsafe {
        let mut buf = vec![0i8; 1024];
        let res = CPXgeterrorstring(env, errcode, buf.as_mut_ptr());
        if res == std::ptr::null_mut() {
            None
        } else {
            Some(String::from_utf8(buf.iter().take_while(|&&i| i != 0 && i != '\n' as i8)
                              .map(|&i| i as u8).collect::<Vec<u8>>()).unwrap())
        }
    }
}

pub struct Env {
    inner: *mut CEnv
}

impl Env {
    pub fn new() -> Result<Env, String> {
        unsafe {
            let mut status = 0;
            let env = CPXopenCPLEX(&mut status);
            if env == std::ptr::null_mut() {
                Err(format!("CPLEX returned NULL for CPXopenCPLEX (status: {})", status))
            } else {
                CPXsetintparam(env, 1035, 1); //ScreenOutput
                CPXsetintparam(env, 1056, 1); //Read_DataCheck
                Ok(Env {
                    inner: env
                })
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Variable<'a> {
    index: Option<usize>,
    ty: VariableType,
    obj: f64,
    lb: f64,
    ub: f64,
    name: &'a str
}

impl<'a> Variable<'a> {
    pub fn new(ty: VariableType, obj: f64, lb: f64, ub: f64, name: &'a str) -> Variable<'a> {
        Variable {
            index: None,
            ty: ty,
            obj: obj,
            lb: lb,
            ub: ub,
            name: name
        }
    }
}

macro_rules! var {
    ($lb:tt <= $name:ident <= $ub:tt -> $obj:tt as $vt:path) => {
        {
            use $crate::VariableType::*;
            Variable::new ($vt, $obj, $lb, $ub, "$name")
        }
    };
    // continuous shorthand
    ($lb:tt <= $name:ident <= $ub:tt -> $obj:tt) => (var!($lb <= $name <= $ub -> $obj as Continuous));
    // omit either lb or ub
    ($lb:tt <= $name:ident -> $obj:tt) => (var!($lb <= $name <= INFINITY -> $obj));
    ($name:ident <= $ub:tt -> $obj:tt) => (var!(0.0 <= $name <= INFINITY -> $obj));
    // omit both
    ($name:ident -> $obj:tt) => (var!(0.0 <= $name -> $obj));

    // typed version
    ($lb:tt <= $name:ident -> $obj:tt as $vt:path) => (var!($lb <= $name <= INFINITY -> $obj as $vt));
    ($name:ident <= $ub:tt -> $obj:tt as $vt:path) => (var!(0.0 <= $name <= INFINITY -> $obj as $vt));
    ($name:ident -> $obj:tt as $vt:path) => (var!(0.0 <= $name -> $obj as $vt));
}

#[derive(Clone, Debug)]
pub struct WeightedVariable {
    var: usize,
    weight: f64
}

impl WeightedVariable {
    pub fn new_var<'a>(var: &'a Variable<'a>, weight: f64) -> Self {
        WeightedVariable {
            var: var.index.unwrap(),
            weight: weight
        }
    }

    pub fn new_idx(idx: usize, weight: f64) -> Self {
        WeightedVariable {
            var: idx,
            weight: weight
        }
    }
}

#[derive(Clone, Debug)]
pub struct Constraint<'a> {
    index: Option<usize>,
    vars: Vec<WeightedVariable>,
    ty: ConstraintType,
    rhs: f64,
    name: &'a str,
}

impl<'a> Constraint<'a> {
    pub fn new(ty: ConstraintType, rhs: f64, name: &'a str) -> Constraint<'a> {
        Constraint {
            index: None,
            vars: vec![],
            ty: ty,
            rhs: rhs,
            name: name
        }
    }

    pub fn add_wvar(&mut self, wvar: WeightedVariable) {
        self.vars.push(wvar)
    }
}

macro_rules! con_ty {
    (=) => ($crate::ConstraintType::Eq);
    (>=) => ($crate::ConstraintType::LessThanEq);
}

macro_rules! con {
    ($name:ident : $rhs:tt $cmp:tt $c1:tt $x1:ident $(+ $c:tt $x:ident)*) => {
        {
            let mut con = Constraint::new(con_ty!($cmp), $rhs, "$name");
            con.add_wvar(WeightedVariable::new_idx($x1, $c1));
            $(
                con.add_wvar(WeightedVariable::new_idx($x, $c));
            )*
            con
        }
    };
}

pub struct Problem<'a> {
    inner: *mut CProblem,
    pub env: &'a Env,
    pub name: &'a str,
    variables: Vec<Variable<'a>>,
    constraints: Vec<Constraint<'a>>
}


#[derive(Clone, Debug)]
pub struct Solution {
    objective: f64,
    variables: Vec<VariableValue>
}

#[derive(Copy, Clone, Debug)]
pub enum ObjectiveType {
    Maximize,
    Minimize
}

#[derive(Copy, Clone, Debug)]
pub enum VariableType {
    Continuous,
    Binary,
    Integer,
    SemiContinuous,
    SemiInteger,
}

#[derive(Copy, Clone, Debug)]
pub enum VariableValue {
    Continuous(f64),
    Binary(bool),
    Integer(i64),
    SemiContinuous(f64),
    SemiInteger(i64)
}


#[derive(Copy, Clone, Debug)]
pub enum ConstraintType {
    LessThanEq,
    Eq,
    GreaterThanEq,
    Ranged,
}

impl VariableType {
    fn to_c(&self) -> c_char {
        match self {
            &VariableType::Continuous => 'C' as c_char,
            &VariableType::Binary => 'B' as c_char,
            &VariableType::Integer => 'I' as c_char,
            &VariableType::SemiContinuous => 'S' as c_char,
            &VariableType::SemiInteger => 'N' as c_char,
        }
    }
}

impl ConstraintType {
    fn to_c(&self) -> c_char {
        match self {
            &ConstraintType::LessThanEq => 'L' as c_char,
            &ConstraintType::Eq => 'E' as c_char,
            &ConstraintType::GreaterThanEq => 'G' as c_char,
            &ConstraintType::Ranged => unimplemented!()
        }
    }
}

impl ObjectiveType {
    fn to_c(&self) -> c_int {
        match self {
            &ObjectiveType::Minimize => 1 as c_int,
            &ObjectiveType::Maximize => -1 as c_int
        }
    }
}

impl<'a> Problem<'a> {
    pub fn new(env: &'a Env, name: &'a str) -> Result<Problem<'a>, String> {
        unsafe {
            let mut status = 0;
            let prob = CPXcreateprob(env.inner, &mut status, CString::new(name).unwrap().as_ptr());
            if prob == std::ptr::null_mut() {
                Err(format!("CPLEX returned NULL for CPXcreateprob ({} ({}))",
                            errstr(env.inner, status).unwrap(), status))
            } else {
                Ok(Problem {
                    inner: prob,
                    env: env,
                    name: name,
                    variables: vec![],
                    constraints: vec![]
                })
            }
        }
    }

    pub fn add_variable(&mut self, var: Variable<'a>)
                        -> Result<usize, String> {
        unsafe {
            let status = CPXnewcols(self.env.inner, self.inner, 1,
                                    &var.obj, &var.lb, &var.ub, match var.ty {
                                        VariableType::Continuous => std::ptr::null(),
                                        _ => &var.ty.to_c()
                                    },
                                    &CString::new(var.name).unwrap().as_ptr());

            if status != 0 {
                Err(format!("Failed to add {:?} variable {} ({} ({}))",
                            var.ty, var.name, errstr(self.env.inner, status).unwrap(), status))
            } else {
                let index = self.variables.len();
                self.variables.push(Variable { index: Some(index), ..var });
                Ok(index)
            }
        }
    }

    pub fn add_constraint(&mut self, con: Constraint<'a>) -> Result<usize, String> {
        let (ind, val): (Vec<i32>, Vec<f64>) = con.vars.iter()
            .filter(|wv| wv.weight != 0.0)
            .map(|wv| (wv.var as i32, wv.weight)).unzip();
        let nz = val.len() as i32;
        unsafe {
            let status = CPXaddrows(self.env.inner, self.inner,
                                    0, 1, nz, &con.rhs,
                                    &con.ty.to_c(), &0, ind.as_ptr(), val.as_ptr(),
                                    std::ptr::null(), &CString::new(con.name).unwrap().as_ptr());

            if status != 0 {
                Err(format!("Failed to add {:?} constraint {} ({} ({}))",
                            con.ty, con.name, errstr(self.env.inner, status).unwrap(), status))
            } else {
                let index = self.constraints.len();
                self.constraints.push(Constraint { index: Some(index), ..con });
                Ok(index)
            }
        }
    }

    pub fn set_objective(&mut self, ty: ObjectiveType, con: Constraint<'a>) -> Result<(), String> {
        let (ind, val): (Vec<i32>, Vec<f64>) = con.vars.iter()
            .map(|wv| (wv.var as i32, wv.weight)).unzip();
        unsafe {
            let status = CPXchgobj(self.env.inner, self.inner,
                                   con.vars.len() as c_int, ind.as_ptr(), val.as_ptr());

            if status != 0 {
                Err(format!("Failed to set objective weights ({} ({}))",
                            errstr(self.env.inner, status).unwrap(), status))
            } else {
                self.set_objective_type(ty)
            }
        }
    }

    pub fn set_objective_type(&mut self, ty: ObjectiveType) -> Result<(), String> {
        unsafe {
            let status = CPXchgobjsen(self.env.inner, self.inner, ty.to_c());
            if status != 0 {
                Err(format!("Failed to set objective type to {:?} ({} ({}))",
                            ty, errstr(self.env.inner, status).unwrap(), status))
            } else {
                Ok(())
            }
        }
    }

    pub fn solve(&mut self) -> Result<Solution, String> {
        // TODO: support multiple solution types...
        unsafe {
            let status = CPXlpopt(self.env.inner, self.inner);
            if status != 0 {
                CPXwriteprob(self.env.inner, self.inner, CString::new("lpex1.lp").unwrap().as_ptr(), std::ptr::null());
                return Err(format!("LP Optimization failed ({} ({}))",
                                   errstr(self.env.inner, status).unwrap(), status));
            }

            let mut objval: f64 = 0.0;
            let status = CPXgetobjval(self.env.inner, self.inner, &mut objval);
            if status != 0 {
                CPXwriteprob(self.env.inner, self.inner, CString::new("lpex1.lp").unwrap().as_ptr(), std::ptr::null());
                return Err(format!("Failed to retrieve objective value ({} ({}))",
                                   errstr(self.env.inner, status).unwrap(), status));
            }

            let mut xs = vec![0f64; self.variables.len()];
            let status = CPXgetx(self.env.inner, self.inner, xs.as_mut_ptr(), 0, self.variables.len() as c_int - 1);
            if status != 0 {
                return Err(format!("Failed to retrieve values for variables ({} ({}))",
                                   errstr(self.env.inner, status).unwrap(), status));
            }

            return Ok(Solution {
                objective: objval,
                variables: xs.iter().zip(self.variables.iter()).map(|(&x, v)| match v.ty {
                    VariableType::Binary => VariableValue::Binary(x == 1.0),
                    VariableType::Continuous => VariableValue::Continuous(x),
                    VariableType::Integer => VariableValue::Integer(x as i64),
                    VariableType::SemiContinuous => VariableValue::SemiContinuous(x),
                    VariableType::SemiInteger => VariableValue::SemiInteger(x as i64),
                }).collect::<Vec<VariableValue>>()
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct() {
        let env = Env::new().unwrap();
        let _prob = Problem::new(&env, "test").unwrap();
    }

    #[test]
    fn add_variable() {
        let env = Env::new().unwrap();
        let mut prob = Problem::new(&env, "test").unwrap();
        prob.add_variable(Variable::new(VariableType::Binary, 1.0, 0.0, 40.0, "x1")).unwrap();
    }

    #[test]
    fn add_constraint() {
        let env = Env::new().unwrap();
        let mut prob = Problem::new(&env, "test").unwrap();
        let var_idx = prob.add_variable(Variable::new(VariableType::Binary, 1.0, 0.0, 40.0, "x1"))
            .unwrap();
        let mut con = Constraint::new(ConstraintType::LessThanEq, 20.0, "c1");
        con.add_wvar(WeightedVariable::new_idx(var_idx, -1.0));
        prob.add_constraint (con).unwrap ();
    }

    #[test]
    fn lpex1() {
        let env = Env::new().unwrap();
        let mut prob = Problem::new(&env, "lpex1").unwrap();
        prob.set_objective_type(ObjectiveType::Maximize).unwrap();
        let x1 = prob.add_variable(var!(0.0 <= x1 <= 40.0 -> 1.0))
            .unwrap();
        let x2 = prob.add_variable(var!(0.0 <= x1 -> 2.0))
            .unwrap();
        let x3 = prob.add_variable(var!(0.0 <= x1 -> 3.0))
            .unwrap();

        prob.add_constraint(con!(c1: 20.0 >= (-1.0) x1 + 1.0 x2 + 1.0 x3)).unwrap();
        prob.add_constraint(con!(c2: 30.0 >= 1.0 x1 + (-3.0) x2 + 1.0 x3)).unwrap();

        let sol = prob.solve().unwrap();
        println!("{:?}", sol);
        assert!(sol.objective == 202.5);
    }
}
