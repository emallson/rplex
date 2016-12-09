extern crate libc;
use libc::{c_int, c_char, c_double, c_long};
use std::ffi::CString;

pub const INFINITY: f64 = 1.0E+20;

enum CEnv {}
enum CProblem {}

type CInt = c_int;

// #[link(name="cplex", kind="static")]
#[allow(non_snake_case)]
#[allow(dead_code)]
extern "C" {
    // instantiation
    fn CPXopenCPLEX(status: *mut c_int) -> *mut CEnv;
    fn CPXcreateprob(env: *mut CEnv, status: *mut c_int, name: *const c_char) -> *mut CProblem;
    fn CPXsetintparam(env: *mut CEnv, param: c_int, value: c_int) -> c_int;
    fn CPXsetdblparam(env: *mut CEnv, param: c_int, value: c_double) -> c_int;
    fn CPXgetintparam(env: *mut CEnv, param: c_int, value: *mut c_int) -> c_int;
    // adding variables and constraints
    fn CPXnewcols(env: *mut CEnv, lp: *mut CProblem, count: CInt,
                   obj: *const c_double, lb: *const c_double, ub: *const c_double,
                   xctype: *const c_char, name: *const *const c_char) -> c_int;
    fn CPXaddrows(env: *mut CEnv, lp: *mut CProblem,
                   col_count: CInt, row_count: CInt, nz_count: CInt,
                   rhs: *const c_double, sense: *const c_char,
                   rmatbeg: *const CInt, rmatind: *const CInt, rmatval: *const c_double,
                   col_name: *const *const c_char, row_name: *const *const c_char) -> c_int;
    // querying
    fn CPXgetnumcols(env: *const CEnv, lp: *mut CProblem) -> CInt;
    // setting objective
    fn CPXchgobj(env: *mut CEnv, lp: *mut CProblem, cnt: CInt, indices: *const CInt, values: *const c_double) -> c_int;
    fn CPXchgobjsen(env: *mut CEnv, lp: *mut CProblem, maxormin: c_int) -> c_int;
    // solving
    fn CPXlpopt(env: *mut CEnv, lp: *mut CProblem) -> c_int;
    fn CPXmipopt(env: *mut CEnv, lp: *mut CProblem) -> c_int;
    // getting solution
    fn CPXgetstat(env: *mut CEnv, lp: *mut CProblem) -> c_int;
    fn CPXgetobjval(env: *mut CEnv, lp: *mut CProblem, objval: *mut c_double) -> c_int;
    fn CPXgetx(env: *mut CEnv, lp: *mut CProblem, x: *mut c_double, begin: CInt, end: CInt) -> c_int;
    fn CPXsolution(env: *mut CEnv, lp: *mut CProblem, lpstat_p: *mut c_int, objval_p: *mut c_double,
                   x: *mut c_double, pi: *mut c_double, slack: *mut c_double, dj: *mut c_double) -> c_int;
    // debugging
    fn CPXgeterrorstring(env: *mut CEnv, errcode: c_int, buff: *mut c_char) -> *mut c_char;
    fn CPXwriteprob(env: *mut CEnv, lp: *mut CProblem, fname: *const c_char, ftype: *const c_char) -> c_int;
    // freeing
    fn CPXcloseCPLEX(env: *const *mut CEnv) -> c_int;
    fn CPXfreeprob(env: *mut CEnv, lp: *const *mut CProblem) -> c_int;
}

fn errstr(env: *mut CEnv, errcode: c_int) -> Result<String, String> {
    unsafe {
        let mut buf = vec![0i8; 1024];
        let res = CPXgeterrorstring(env, errcode, buf.as_mut_ptr());
        if res == std::ptr::null_mut() {
            Err(format!("No error string for {}", errcode))
        } else {
            Ok(String::from_utf8(buf.iter().take_while(|&&i| i != 0 && i != '\n' as i8)
                                 .map(|&i| i as u8).collect::<Vec<u8>>()).unwrap())
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum ParamType {
    Integer(c_int),
    Double(c_double),
    Boolean(c_int)
}

#[derive(Copy, Clone, Debug)]
pub enum EnvParam {
    Threads(u64),
    ScreenOutput(bool),
    RelativeGap(f64),
}

impl EnvParam {

    fn to_id(&self) -> c_int {
        use EnvParam::*;
        match self {
            &Threads(_) => 1067,
            &ScreenOutput(_) => 1035,
            &RelativeGap(_) => 2009,
        }
    }

    fn param_type(&self) -> ParamType {
        use EnvParam::*;
        use ParamType::*;
        match self {
            &Threads(t) => Integer(t as c_int),
            &ScreenOutput(b) => Boolean(b as c_int),
            &RelativeGap(g) => Double(g as c_double),
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
                // CPXsetintparam(env, 1035, 1); //ScreenOutput
                // CPXsetintparam(env, 1056, 1); //Read_DataCheck
                Ok(Env {
                    inner: env
                })
            }
        }
    }

    pub fn set_param(&mut self, p: EnvParam) -> Result<(), String> {
        unsafe {
            let status = match p.param_type() {
                ParamType::Integer(i) => CPXsetintparam(self.inner, p.to_id(), i),
                ParamType::Boolean(b) => CPXsetintparam(self.inner, p.to_id(), b),
                ParamType::Double(d) => CPXsetdblparam(self.inner, p.to_id(), d),
            };

            if status != 0 {
                return match errstr(self.inner, status) {
                    Ok(s) => Err(s),
                    Err(e) => Err(e)
                };
            } else {
                return Ok(());
            }
        }
    }

}

impl Drop for Env {
    fn drop(&mut self) {
        unsafe {
            assert!(CPXcloseCPLEX (&self.inner) == 0);
        }
    }
}

#[derive(Clone, Debug)]
pub struct Variable {
    index: Option<usize>,
    ty: VariableType,
    obj: f64,
    lb: f64,
    ub: f64,
    name: String
}

impl Variable {
    pub fn new<S>(ty: VariableType, obj: f64, lb: f64, ub: f64, name: S) -> Variable
        where S: Into<String> {
        Variable {
            index: None,
            ty: ty,
            obj: obj,
            lb: lb,
            ub: ub,
            name: name.into()
        }
    }
}

#[macro_export]
macro_rules! var {
    ($lb:tt <= $name:tt <= $ub:tt -> $obj:tt as $vt:path) => {
        {
            use $crate::VariableType::*;
            Variable::new ($vt, $obj, $lb, $ub, $name)
        }
    };
    // continuous shorthand
    ($lb:tt <= $name:tt <= $ub:tt -> $obj:tt) => (var!($lb <= $name <= $ub -> $obj as Continuous));
    // omit either lb or ub
    ($lb:tt <= $name:tt -> $obj:tt) => (var!($lb <= $name <= INFINITY -> $obj));
    ($name:tt <= $ub:tt -> $obj:tt) => (var!(0.0 <= $name <= INFINITY -> $obj));
    // omit both
    ($name:tt -> $obj:tt) => (var!(0.0 <= $name -> $obj));

    // typed version
    ($lb:tt <= $name:tt -> $obj:tt as $vt:path) => (var!($lb <= $name <= INFINITY -> $obj as $vt));
    ($name:tt <= $ub:tt -> $obj:tt as $vt:path) => (var!(0.0 <= $name <= INFINITY -> $obj as $vt));
    ($name:tt -> $obj:tt as Binary) => (var!(0.0 <= $name <= 1.0 -> $obj as Binary));
    ($name:tt -> $obj:tt as $vt:path) => (var!(0.0 <= $name -> $obj as $vt));
}

#[derive(Clone, Debug)]
pub struct WeightedVariable {
    var: usize,
    weight: f64
}

impl WeightedVariable {
    pub fn new_var(var: &Variable, weight: f64) -> Self {
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
pub struct Constraint {
    index: Option<usize>,
    vars: Vec<WeightedVariable>,
    ty: ConstraintType,
    rhs: f64,
    name: String
}

impl Constraint {
    pub fn new<S, F>(ty: ConstraintType, rhs: F, name: S) -> Constraint
        where S: Into<String>,
              F: Into<f64> {
        Constraint {
            index: None,
            vars: vec![],
            ty: ty,
            rhs: rhs.into(),
            name: name.into()
        }
    }

    pub fn add_wvar(&mut self, wvar: WeightedVariable) {
        self.vars.push(wvar)
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! con_ty {
    (=) => ($crate::ConstraintType::Eq);
    (>=) => ($crate::ConstraintType::LessThanEq);
    (<=) => ($crate::ConstraintType::GreaterThanEq);
}

/// Macro to simplify writing constraints. In the future, this should
/// be made recursive to improve flexibility rather than repeatedly
/// adding special cases.
#[macro_export]
macro_rules! con {
    ($name:tt : $rhs:tt $cmp:tt sum $body:expr) => {
        {
            let mut con = Constraint::new(con_ty!($cmp), $rhs, $name);
            for &var in $body {
                con.add_wvar(WeightedVariable::new_idx(var, 1.0));
            }
            con
        }
    };
    ($name:tt : $rhs:tt $cmp:tt wsum $body:expr) => {
        {
            let mut con = Constraint::new(con_ty!($cmp), $rhs, $name);
            for (var, weight) in $body {
                con.add_wvar(WeightedVariable::new_idx(var, weight));
            }
            con
        }
    };
    ($name:tt : $rhs:tt $cmp:tt $c1:tt $x1:ident $(+ $c:tt $x:ident)*) => {
        {
            let mut con = Constraint::new(con_ty!($cmp), $rhs, $name);
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
    pub name: String,
    variables: Vec<Variable>,
    constraints: Vec<Constraint>
}


#[derive(Clone, Debug)]
pub struct Solution {
    pub objective: f64,
    pub variables: Vec<VariableValue>
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

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum VariableValue {
    Continuous(f64),
    Binary(bool),
    Integer(CInt),
    SemiContinuous(f64),
    SemiInteger(CInt)
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
    pub fn new<S>(env: &'a Env, name: S) -> Result<Self, String>
        where S: Into<String> {
        unsafe {
            let mut status = 0;
            let name = name.into();
            let prob = CPXcreateprob(env.inner, &mut status, CString::new(name.as_str()).unwrap().as_ptr());
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

    pub fn add_variable(&mut self, var: Variable)
                        -> Result<usize, String> {
        unsafe {
            let status = CPXnewcols(self.env.inner, self.inner, 1,
                                    &var.obj, &var.lb, &var.ub, &var.ty.to_c(),
                                    &CString::new(var.name.as_str()).unwrap().as_ptr());

            if status != 0 {
                Err(format!("Failed to add {:?} variable {} ({} ({}))",
                            var.ty, var.name, errstr(self.env.inner, status).unwrap(), status))
            } else {
                let index = CPXgetnumcols(self.env.inner, self.inner) as usize - 1;
                self.variables.push(Variable { index: Some(index), ..var });
                Ok(index)
            }
        }
    }

    pub fn add_constraint(&mut self, con: Constraint) -> Result<usize, String> {
        let (ind, val): (Vec<CInt>, Vec<f64>) = con.vars.iter()
            .filter(|wv| wv.weight != 0.0)
            .map(|wv| (wv.var as CInt, wv.weight)).unzip();
        let nz = val.len() as CInt;
        unsafe {
            let status = CPXaddrows(self.env.inner, self.inner,
                                    0, 1, nz, &con.rhs,
                                    &con.ty.to_c(), &0, ind.as_ptr(), val.as_ptr(),
                                    std::ptr::null(), &CString::new(con.name.as_str()).unwrap().as_ptr());

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

    pub fn set_objective(&mut self, ty: ObjectiveType, con: Constraint) -> Result<(), String> {
        let (ind, val): (Vec<CInt>, Vec<f64>) = con.vars.iter()
            .map(|wv| (wv.var as CInt, wv.weight)).unzip();
        unsafe {
            let status = CPXchgobj(self.env.inner, self.inner,
                                   con.vars.len() as CInt, ind.as_ptr(), val.as_ptr());

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

    pub fn write<S>(&self, name: S) -> Result<(), String>
        where S: Into<String>{
        unsafe {
            let status = CPXwriteprob(self.env.inner, self.inner,
                                      CString::new(name.into().as_str()).unwrap().as_ptr(),
                                      std::ptr::null());
            if status != 0 {
                return match errstr(self.env.inner, status) {
                    Ok(s) => Err(s),
                    Err(e) => Err(e)
                };
            } else {
                Ok(())
            }
        }
    }

    pub fn solve(&mut self) -> Result<Solution, String> {
        // TODO: support multiple solution types...
        unsafe {
            let status = CPXmipopt(self.env.inner, self.inner);
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
            let status = CPXgetx(self.env.inner, self.inner, xs.as_mut_ptr(), 0, self.variables.len() as CInt - 1);
            if status != 0 {
                return Err(format!("Failed to retrieve values for variables ({} ({}))",
                                   errstr(self.env.inner, status).unwrap(), status));
            }

            return Ok(Solution {
                objective: objval,
                variables: xs.iter().zip(self.variables.iter()).map(|(&x, v)| match v.ty {
                    VariableType::Binary => VariableValue::Binary(x == 1.0),
                    VariableType::Continuous => VariableValue::Continuous(x),
                    VariableType::Integer => VariableValue::Integer(x as CInt),
                    VariableType::SemiContinuous => VariableValue::SemiContinuous(x),
                    VariableType::SemiInteger => VariableValue::SemiInteger(x as CInt),
                }).collect::<Vec<VariableValue>>()
            });
        }
    }
}

impl<'a> Drop for Problem<'a> {
    fn drop(&mut self) {
        unsafe { assert!(CPXfreeprob (self.env.inner, &self.inner) == 0); }
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
        let x1 = prob.add_variable(var!(0.0 <= "x1" <= 40.0 -> 1.0)).unwrap();
        let x2 = prob.add_variable(var!("x2" -> 2.0)).unwrap();
        let x3 = prob.add_variable(var!("x3" -> 3.0)).unwrap();
        println!("{} {} {}", x1, x2, x3);

        prob.add_constraint(con!("c1": 20.0 >= (-1.0) x1 + 1.0 x2 + 1.0 x3)).unwrap();
        prob.add_constraint(con!("c2": 30.0 >= 1.0 x1 + (-3.0) x2 + 1.0 x3)).unwrap();

        prob.write("lpex1_test.lp").unwrap();
        let sol = prob.solve().unwrap();
        println!("{:?}", sol);
        assert!(sol.objective == 202.5);
        assert!(sol.variables == vec![VariableValue::Continuous(40.0),
                                      VariableValue::Continuous(17.5),
                                      VariableValue::Continuous(42.5)]);
    }

    #[test]
    #[ignore]
    fn set_param() {
        let mut _env = Env::new().unwrap();
        // this is perhaps why not to use tuple structs as the enum
        // variants for params...
        // assert!(env.get_param(EnvParam::ScreenOutput(false)).unwrap() == false);
        // env.set_param(EnvParam::ScreenOutput(true)).unwrap();
        // assert!(env.get_param(EnvParam::ScreenOutput(false)).unwrap() == true);
    }
}
