class ContactsService:
    def __init__(self):
        self._contacts = []

    def add_contact(self, contact):
        self._contacts.append(contact)

    def get_contact(self, payload):
        for contact in self._contacts:
            if payload == contact.name:
                return contact
                break

    def get_contacts(self):
        return self._contacts

    def del_contact(self, payload):
        for contact in self._contacts:
            if payload == contact.name:
                self._contacts.remove(contact)
                break
