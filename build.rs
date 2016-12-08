extern crate glob;
use glob::glob;

fn main() {

    let env_path = option_env!("CPLEX_LIB");

    if let Some(cplex_path) = env_path {
        println!("cargo:rustc-link-search=native={}", cplex_path);
    } else {
        for path in glob("/opt/ibm/ILOG/*/cplex/lib/*/static_pic").unwrap() {
            match path {
                Ok(libdir) => println!("cargo:rustc-link-search=native={}", libdir.display()),
                Err(e) => println!("{:?}", e)
            }
        }
    }

    println!("cargo:rustc-link-lib=static=cplex");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
}
