import asyncio
from playwright.async_api import async_playwright

# Microsoft (MSFT) 2017 se 2026 tak ki filings
TARGET_FILINGS = [
    {
        "company": "MSFT",
        "year": "2017",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000156459017014900/msft-10k_20170630.htm"
    },
    {
        "company": "MSFT",
        "year": "2018",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000156459018019062/msft-10k_20180630.htm"
    },
    {
        "company": "MSFT",
        "year": "2019",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000156459019027952/msft-10k_20190630.htm"
    },
    {
        "company": "MSFT",
        "year": "2020",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000156459020034944/msft-10k_20200630.htm"
    },
    {
        "company": "MSFT",
        "year": "2021",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000156459021039151/msft-10k_20210630.htm"
    },
    {
        "company": "MSFT",
        "year": "2022",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000156459022026876/msft-10k_20220630.htm"
    },
    {
        "company": "MSFT",
        "year": "2023",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm"
    },
    {
        "company": "MSFT",
        "year": "2024",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm"
    },
    {
        "company": "MSFT",
        "year": "2025",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000095017025100235/msft-20250630.htm"
    },
    {
        "company": "MSFT",
        "year": "2026",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000119312526323660/msft-20260630.htm"
    }
]

async def download_filing(playwright, filing):
    request_context = await playwright.request.new_context(
        extra_http_headers={
            "User-Agent": "Hifza UniversityStudent hifza@example.com",
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov"
        }
    )
    
    print(f"Fetching {filing['company']} ({filing['year']})...")
    response = await request_context.get(filing['url'])
    
    if response.status != 200:
        print(f"Failed with status: {response.status} for {filing['year']}")
        await request_context.dispose()
        return

    html_content = await response.text()
    
    # Render HTML string into clean PDF via Headless Browser
    browser = await playwright.chromium.launch(headless=True)
    page = await browser.new_page()
    
    # Load HTML directly without hitting SEC network again
    await page.set_content(html_content, wait_until="domcontentloaded")
    
    output_filename = f"{filing['company']}_{filing['year']}_10K.pdf"
    await page.pdf(
        path=output_filename,
        format="A4",
        print_background=True,
        margin={"top": "0.3in", "bottom": "0.3in", "left": "0.3in", "right": "0.3in"}
    )
    
    print(f"Successfully saved clean PDF: {output_filename}")
    await browser.close()
    await request_context.dispose()

async def main():
    async with async_playwright() as playwright:
        for filing in TARGET_FILINGS:
            await download_filing(playwright, filing)

if __name__ == "__main__":
    asyncio.run(main())