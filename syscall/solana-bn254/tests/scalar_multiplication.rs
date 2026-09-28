//! Direct checks of scalar Montgomery multiplication against arkworks.
//!
//! Inputs are raw, reduced integers. Expected outputs are a * b * R^-1
//! modulo the scalar field modulus, where R = 2^256.
//!
//! Neither input construction nor reference calculations use the backend's
//! Montgomery conversion functions.

use ark_bn254::Fr as ArkFr;
use ark_ff::{BigInt, Field, PrimeField};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use solana_bn254::backend::{Backend, Fr, MontgomeryBackend, U256};

type B = Backend<Fr>;

fn from_ark(value: ArkFr) -> U256 {
    U256::new(value.into_bigint().0)
}

fn as_ark_integer(value: &U256) -> ArkFr {
    ArkFr::from_bigint(BigInt::<4>(value.0))
        .expect("test operand must be smaller than the scalar field modulus")
}

fn montgomery_r_inverse() -> ArkFr {
    ArkFr::from(2u64)
        .pow([256u64])
        .inverse()
        .expect("R must be invertible modulo the scalar field modulus")
}

fn random_operand(rng: &mut StdRng) -> U256 {
    let bytes = rng.random::<[u8; 32]>();
    from_ark(ArkFr::from_le_bytes_mod_order(&bytes))
}

fn assert_product(a: &U256, b: &U256, r_inverse: ArkFr) {
    let expected = from_ark(as_ark_integer(a) * as_ark_integer(b) * r_inverse);
    let actual = B::mul(a, b);

    // Compare raw limbs so an unreduced result cannot pass by being
    // normalized during conversion back into an arkworks field element.
    assert_eq!(actual, expected, "incorrect product for a={a:?}, b={b:?}");
}

fn boundary_operands() -> Vec<U256> {
    let mut values = vec![
        U256::zero(),
        U256::one(),
        U256::new([2, 0, 0, 0]),
        U256::new([3, 0, 0, 0]),
    ];

    // Values immediately below the modulus, including a subtraction
    // that borrows across the lowest limb.
    for delta in [1u64, 2, 3, 4, 7, 8, 15, 16, u64::MAX] {
        values.push(from_ark(-ArkFr::from(delta)));
    }

    // Values around limb boundaries exercise carry propagation through
    // one, two, and three lower limbs. Include the highest usable bits.
    let one = ArkFr::from(1u64);
    for bit in [1usize, 63, 64, 65, 127, 128, 129, 191, 192, 193, 252, 253] {
        let mut limbs = [0u64; 4];
        limbs[bit / 64] = 1u64 << (bit % 64);

        let power = as_ark_integer(&U256::new(limbs));
        values.push(from_ark(power - one));
        values.push(from_ark(power));
        values.push(from_ark(power + one));
    }

    values
}

#[test]
fn mul_boundary_values_match_arkworks() {
    let r_inverse = montgomery_r_inverse();
    let values = boundary_operands();

    for a in &values {
        for b in &values {
            assert_product(a, b, r_inverse);
        }
    }
}

#[test]
fn mul_seeded_inputs_match_arkworks() {
    let mut rng = StdRng::seed_from_u64(0x6d75_6c5f_7061_6972);
    let r_inverse = montgomery_r_inverse();

    for _ in 0..4096 {
        let a = random_operand(&mut rng);
        let b = random_operand(&mut rng);
        assert_product(&a, &b, r_inverse);
    }
}

#[test]
fn mul_chains_match_arkworks() {
    let mut rng = StdRng::seed_from_u64(0x6d75_6c5f_6368_6169);
    let r_inverse = montgomery_r_inverse();

    for chain in 0..64 {
        let mut actual = random_operand(&mut rng);
        let mut expected = as_ark_integer(&actual);

        for step in 0..64 {
            let factor = random_operand(&mut rng);

            actual = B::mul(&actual, &factor);
            expected *= as_ark_integer(&factor) * r_inverse;

            assert_eq!(
                actual,
                from_ark(expected),
                "incorrect product in chain {chain}, step {step}"
            );
        }
    }
}

fn assert_sum_of_products<const N: usize>(a: &[U256; N], b: &[U256; N], r_inverse: ArkFr) {
    let sum: ArkFr = a
        .iter()
        .zip(b)
        .map(|(x, y)| as_ark_integer(x) * as_ark_integer(y))
        .sum();
    assert_eq!(
        B::sum_of_products(a, b),
        from_ark(sum * r_inverse),
        "incorrect sum of products for a={a:?}, b={b:?}"
    );
}

fn check_sum_of_products<const N: usize>(rng: &mut StdRng, r_inverse: ArkFr) {
    let values = boundary_operands();
    for start in 0..values.len() {
        let a: [U256; N] = core::array::from_fn(|k| values[(start + k) % values.len()]);
        let b: [U256; N] = core::array::from_fn(|k| values[(start + 3 * k + 1) % values.len()]);
        assert_sum_of_products(&a, &b, r_inverse);
    }

    // Operands just below MODULUS push each chunk sum toward `MODULUS * 2^256`.
    let operands: [fn(&mut StdRng) -> U256; 2] = [random_operand, |rng| {
        from_ark(-ArkFr::from(u64::from(rng.random::<u32>()) + 1))
    }];
    for operand in operands {
        for _ in 0..64 {
            let a: [U256; N] = core::array::from_fn(|_| operand(rng));
            let b: [U256; N] = core::array::from_fn(|_| operand(rng));
            assert_sum_of_products(&a, &b, r_inverse);
        }
    }
}

#[test]
fn sum_of_products_matches_arkworks() {
    let mut rng = StdRng::seed_from_u64(0x7375_6d5f_7072_6f64);
    let r_inverse = montgomery_r_inverse();
    macro_rules! widths {
        ($($n:literal)+) => { $(check_sum_of_products::<$n>(&mut rng, r_inverse);)+ };
    }
    widths!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17);
}

#[test]
fn sum_of_products_small_modulus() {
    struct SmallField;

    impl solana_bn254::backend::Field for SmallField {
        const MODULUS: U256 = U256::new([97, 0, 0, 0]);
        const INV: u64 = 0x5c5f02a3a0fd5c5f;
        const R2: U256 = U256::new([35, 0, 0, 0]);
    }

    type SmallBackend = Backend<SmallField>;
    assert_eq!(SmallBackend::sum_of_products(&[], &[]), U256::zero());

    let radix = (0..256).fold(1u64, |r, _| 2 * r % 97);
    let r_inverse = (1..97).find(|r| r * radix % 97 == 1).unwrap();
    for a in 0..97 {
        for b in 0..97 {
            let x = U256::new([a, 0, 0, 0]);
            let y = U256::new([b, 0, 0, 0]);
            for (actual, terms) in [
                (SmallBackend::sum_of_products(&[x], &[y]), 1),
                (SmallBackend::sum_of_products(&[x; 2], &[y; 2]), 2),
            ] {
                let expected = U256::new([terms * a * b * r_inverse % 97, 0, 0, 0]);
                assert_eq!(actual, expected, "a={a}, b={b}, terms={terms}");
            }
        }
    }
}
