import os
import asyncio
from playwright.async_api import async_playwright

ROOT_DIR = r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results\AE) Landmark_RMST"

async def convert_html_to_png(browser, html_path):
    # Strictly change .html to .png for the output path
    base_path = os.path.splitext(html_path)[0]
    png_path = base_path + ".png"
    
    page = await browser.new_page(device_scale_factor=2)
    
    try:
        # Convert path to file URL
        file_url = f"file:///{os.path.abspath(html_path).replace('\\', '/')}"
        
        # Open page and wait for content
        await page.goto(file_url, wait_until="networkidle")
        await asyncio.sleep(2) # Extra time for Plotly animations
        
        # Take screenshot
        await page.screenshot(path=png_path, full_page=True)
        print(f"Created PNG: {png_path}")
        
    except Exception as e:
        print(f"Failed {html_path}: {e}")
    finally:
        await page.close()

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        for root, dirs, files in os.walk(ROOT_DIR):
            for file in files:
                # Only process actual .html files
                if file.lower().endswith(".html"):
                    html_full_path = os.path.join(root, file)
                    await convert_html_to_png(browser, html_full_path)
        
        await browser.close()
        print("\n--- Conversion Complete ---")

if __name__ == "__main__":
    asyncio.run(main())