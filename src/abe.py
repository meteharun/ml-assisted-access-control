from charm.toolbox.pairinggroup import PairingGroup, GT
from charm.schemes.abenc.ac17 import AC17CPABE


class ABE:
    """Abstracts ABE setup, key generation, encryption, and decryption."""

    def __init__(
        self,
        scheme=AC17CPABE,
        pairing_group=PairingGroup("MNT224"),
        assumption_size=2,
    ):
        self.pairing_group  = pairing_group
        self.assumption_size = assumption_size
        self.cpabe          = scheme(self.pairing_group, self.assumption_size)
        (self.pk, self.msk) = self.cpabe.setup()

    def get_random_plaintext(self):
        return self.pairing_group.random(GT)

    def generate_key(self, attribute_list: list):
        """Generate an ABE decryption key bound to the given attribute list."""
        return self.cpabe.keygen(self.pk, self.msk, attribute_list)

    def encrypt(self, plaintext, policy_string: str):
        """Encrypt plaintext under an ABE access policy.

        Args:
            plaintext:      A pairing group element (GT) to encrypt.
            policy_string:  E.g. "SCORE3 or SCORE4"

        Returns:
            ABE ciphertext bound to the policy.
        """
        return self.cpabe.encrypt(self.pk, plaintext, policy_string)

    def decrypt(self, key, ciphertext):
        """Attempt to decrypt ciphertext with key.

        Returns the plaintext if the key satisfies the policy, False otherwise.
        """
        return self.cpabe.decrypt(self.pk, ciphertext, key)


# ── Smoke-test ────────────────────────────────────────────────────────────────

def main():
    abe = ABE()

    attribute_lists = [
        ["MANAGER", "NOKIA"],
        ["MANAGER", "OTHER"],
        ["RESEARCHER", "NOKIA"],
        ["DEVELOPER", "NOKIA"],
    ]
    keys = [abe.generate_key(attrs) for attrs in attribute_lists]

    msg        = abe.pairing_group.random(GT)
    policy_str = "(NOKIA and (RESEARCHER OR MANAGER))"
    ctxt       = abe.encrypt(msg, policy_str)
    print(f"Policy: {policy_str}\n")

    for key in keys:
        result = abe.decrypt(key, ctxt)
        status = "✓ SUCCESS" if result == msg else "✗ FAILED (expected)"
        print(f"  {status}  attributes={key['attr_list']}")


if __name__ == "__main__":
    main()
