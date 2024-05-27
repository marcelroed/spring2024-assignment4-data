import re

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+')
def mask_emails(text: str):
    """Replace all email addresses with |||EMAIL_ADDRESS|||, return the new string and the number of instances that were masked."""
    return EMAIL_RE.sub('|||EMAIL_ADDRESS|||', text), len(EMAIL_RE.findall(text))

PHONE_NUMBER_RE = re.compile(r'\(?\d{3}\)?[- ]?\d{3}[ -]?\d{4}')
def mask_phone_numbers(text: str):
    """Replace all phone numbers with |||PHONE_NUMBER|||, return the new string and the number of instances that were masked."""
    return PHONE_NUMBER_RE.sub('|||PHONE_NUMBER|||', text), len(PHONE_NUMBER_RE.findall(text))

IP_ADDRESS_RE = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
def mask_ips(text: str):
    """Replace all IP addresses with |||IP_ADDRESS|||, return the new string and the number of instances that were masked."""
    return IP_ADDRESS_RE.sub('|||IP_ADDRESS|||', text), len(IP_ADDRESS_RE.findall(text))


def main():
    from cs336_data.extract_text import iterwarc, extract_text_from_html_bytes

    count = 0

    for record in iterwarc('data/CC-MAIN-20180420081400-20180420101400-00118.warc.gz'):
        text = extract_text_from_html_bytes(record.reader.read())
        email_masked, email_count = mask_emails(text)
        phone_number_masked, phone_number_count = mask_phone_numbers(text)
        ip_addr_masked, ip_addr_count = mask_ips(text)
        if email_count > 0 or phone_number_count > 0 or ip_addr_count > 0:
            print(f'>>> Masked out {email_count} emails, {phone_number_count} phone numbers, and {ip_addr_count} IP addresses:')
            print(email_masked)
            print(phone_number_masked)
            print(ip_addr_masked)
            print()
            count += 1
            if count >= 20:
                break
        



if __name__ == '__main__':
    main()