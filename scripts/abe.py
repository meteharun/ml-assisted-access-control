from charm.toolbox.pairinggroup import PairingGroup, GT
from charm.schemes.abenc.ac17 import AC17CPABE

class ABE:
    """ Abstracts the ABE details (setup, key generation, encryption and decryption)
    """
    def __init__(self, scheme=AC17CPABE, pairing_group=PairingGroup('MNT224'), assumption_size=2):
        """Initialize our setup

        Args:
            scheme (ABEnc, optional): The ABE scheme to be used. It comes from the ABE libray and can be: . Defaults to AC17CPABE.
            pairing_group (_type_, optional): _description_. Defaults to PairingGroup('MNT224').
            assumption_size (int, optional): _description_. Defaults to 2.
        """
        self.pairing_group = pairing_group
        self.assumption_size = assumption_size
        self.cpabe = scheme(self.pairing_group, self.assumption_size)
        (self.pk, self.msk) = self.cpabe.setup()
        
    def get_random_plaintext(self):
        return self.pairing_group.random(GT)
        
    def generate_key(self, attribute_list):
        """ Generates an ABE key

        Args:
            attribute_list ([str]): list of attributes to associate to the key

        Returns:
            _type_: Returns an ABE key with the associated attributes
        """
        return self.cpabe.keygen(self.pk, self.msk, attribute_list)
    
    def encrypt(self, plaintext, policy_string):
        """Encrypts a plaintext using ABE with the given policy string

        Args:
            plaintext (string): Plaintext to encrypt
            policy_string (string): string with attribute policy that will allow decryption of the plaintext. Example: "attribute1 and attribute2"

        Returns:
            _type_: Returns encrypted ciphertext with the policy
        """
        return self.cpabe.encrypt(self.pk, plaintext, policy_string)
    
    def decrypt(self, key, ciphertext):
        """Attempts to decrypt a ciphertext using the given key.

        Args:
            key (_type_): ABE Key used to decrypt the ciphertext
            ciphertext (_type_): Ciphertext to decrypt

        Returns:
            string: It returns a decrypted plaintext if the key satisfies the policy
        """
        plaintext = self.cpabe.decrypt(self.pk, ciphertext, key)
        return plaintext


def main():
    # Create an ABE encryption object using the default values for scheme, pairing group and assumption size.
    abe = ABE()
    
    attribute_lists = [['MANAGER', 'NOKIA'], ['MANAGER', 'OTHER'], ['RESEARCHER', 'NOKIA'], ['DEVELOPER', 'NOKIA']]
    keys = []

    for attribute_list in attribute_lists:
        key = abe.generate_key(attribute_list)
        keys.append(key)
        
    # choose a random message
    msg = abe.pairing_group.random(GT)

    # generate a ciphertext with a policy that says only NOKIA MANAGER can decrypt the ciphertext
    policy_str = '(NOKIA and (RESEARCHER OR MANAGER))'
    ctxt = abe.encrypt(msg, policy_str)
    print(f"Ciphertext policy is: {policy_str}")
    
    # print("Try to decrypt with all the keys we have:")
    for key in keys:
        print(f"Try to decrypt ciphertext with key with attributes: {key['attr_list']}")
        dec_msg = abe.decrypt(key, ctxt)
        if (dec_msg == msg):
            print(f"Decrypted message: \n{dec_msg}\nis the same as original message \n{msg}")
            print("Decryption succeeded")
        else:
            print("Decryption failed")

if __name__ == "__main__":
    debug = True
    main()
