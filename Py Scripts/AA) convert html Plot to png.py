import os
import asyncio
from playwright.async_api import async_playwright

# Set your root directory here
ROOT_DIR = r"C:\github\CzechFOI-DRATE-OPENSCI\Plot Results"

async def convert_html_to_png(browser, html_path):
    # Determine the output path (same name, .png extension)
    png_path = os.path.splitext(html_path)[0] + ".png"
    
    # Skip if PNG already exists to save time (optional)
    # if os.path.exists(png_path): return

    page = await browser.new_page(device_scale_factor=2) # 2x scale for high resolution
    
    try:
        # Convert absolute path to file URL
        file_url = f"file:///{os.path.abspath(html_path).replace('\\', '/')}"
        
        await page.goto(file_url, wait_until="networkidle")
        
        # Give Plotly/Charts an extra second to animate into place
        await asyncio.sleep(1.5)
        
        # Capture the plot. full_page=True handles varying chart sizes.
        await page.screenshot(path=png_path, full_page=True)
        print(f"Captured: {png_path}")
        
    except Exception as e:
        print(f"Failed to convert {html_path}: {e}")
    finally:
        await page.close()

async def main():
    async with async_playwright() as p:
        # Launch headless browser
        browser = await p.chromium.launch(headless=True)
        
        # Walk through all subdirectories
        tasks = []
        for root, dirs, files in os.walk(ROOT_DIR):
            for file in files:
                if file.lower().endswith(".html"):
                    html_full_path = os.path.join(root, file)
                    # Process files one by one to avoid memory overload with many tabs
                    await convert_html_to_png(browser, html_full_path)
        
        await browser.close()
        print("\n--- Processing Complete ---")

if __name__ == "__main__":
    asyncio.run(main())