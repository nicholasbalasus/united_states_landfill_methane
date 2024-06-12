# This script contains functions for scraping the EPA FLIGHT tool. It is
# set up for the version containing up to 2022 data. This script will break
# when the future FLIGHT data is released (but should be easily adaptable).

import re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

def recurisve_requests(url, tries=0):
    """ Runs requests.get(url) until it works up to 50 tries
    Parameters
    ----------
    url : str
        The web page the data should be retrieved from
    tries : int
        How many times the request for this url has timed out already

    Returns
    -------
    requests.models.Response
        Response from the web page
    """
    
    if tries > 50:
        raise Exception(f"Too many tries for url = {url}!")
    try:
        response = requests.get(url, timeout=5)
        return response
    except:
        return recurisve_requests(url, tries+1)
    
def scrape_flight_ghgrp(id, flightXLS="../resources/flight.xls", verbose=False):
    """Scrapes FLIGHT database to get information about a given GHGRP ID
    Parameters
    ----------
    id : int
        GHGRP ID to gather data about
    flightXLS : str
        Path to a flight.xls file summarizing the yearly GHGRP
    verbose : bool
        Whether to print warnings about missing data

    Returns
    -------
    pandas.core.frame.DataFrame
        DataFrame with info about the facility for every year it reported
    """

    # Use the FLIGHT summary sheet to see which years this landfill reported
    # (1) go to the flight tool: https://ghgdata.epa.gov/ghgp/main.do
    # (2) filter to only "Municipal Landfills" sector and only "CH4" GHG
    # (3) export data for "All Reporting Years" and download "flight.xls"
    years = []
    for year in range(2010,2023):
        summary_sheet = pd.read_excel(flightXLS, header=6, sheet_name=str(year))
        if len(summary_sheet.loc[summary_sheet["GHGRP ID"] == id]) == 1:
            years.append(year)

    # Initialize DataFrame
    df = pd.DataFrame({"company": pd.Series((), dtype="str")})

    # For each year, scrape the data about this facility
    for idx,year in enumerate(years):

        # Start a new row in the DataFrame
        df.loc[idx,"year"] = int(year)

        # Read FLIGHT sheet to get high-level information
        summary_sheet = pd.read_excel(flightXLS, header=6,
                                      sheet_name=str(year))
        subset = summary_sheet.loc[summary_sheet["GHGRP ID"] == id]
        assert len(subset) == 1
        df.loc[idx,"company"] = subset.iloc[0]["PARENT COMPANIES"]
        df.loc[idx,"name"] = subset.iloc[0]["FACILITY NAME"]
        df.loc[idx,"latitude"] = subset.iloc[0]["LATITUDE"]
        df.loc[idx,"longitude"] = subset.iloc[0]["LONGITUDE"]

        # Read first page of FLIGHT to get the following variables:
        # [Emis_Reported, Emis_Forward, Emis_Backward, Gas_Collection]
        url = (f"https://ghgdata.epa.gov/ghgp/service/facilityDetail/"
               f"{year}?id={id}&ds=E&et=&popup=true")
        response = recurisve_requests(url)
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
        data =  {}
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) == 2:
                key = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                data[key] = value
                    
        key = "Does the landfill have an active gas collection system?"
        if key in data.keys():
            df.loc[idx,"Gas_Collection"] = data[key] == "Y"
        else:
            df.loc[idx,"Gas_Collection"] = False
            if verbose:
                print(f"Missing {key} field for {id} & {year}... assuming N")

        df.loc[idx,"Emis_Reported"] = float(
            data["Municipal Landfills"].replace(",",""))
        labels = ["Emis_Forward", "Emis_Backward"]
        begin = "Landfill emissions estimated from "
        ends = ["modeled methane generation and other factors",
                "methane recovery, destruction and other factors"]
        keys = [f"{begin}{e}" for e in ends]
        for label,key in zip(labels,keys):
            if key in data.keys():
                df.loc[idx,label] = float(data[key].replace(",",""))
            else:
                df.loc[idx,label] = np.nan

        # Parse a deeper page of FLIGHT to get the following variables
        # [Recovered, Collection_Eff, If_Gen_Rec_0]
        # This page is a mess since earlier years don't have every field
        # This is why there are a lot of "if" statements here
        url = (f"https://ghgdata.epa.gov/ghgp/service/html/"
               f"{year}?id={id}&et=undefined")
        soup = BeautifulSoup(recurisve_requests(url).text, "html.parser")
        data = {}
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) == 2:
                key = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                data[key] = value

        # These are only variables we can get if gas collecting
        if "Estimated Gas Collection Efficiency HH3" in data.keys():

            key = "Annual Quantity Of Recovered MethaneHH4"
            if data[key].split("\n")[0] == "(":
                if verbose:
                    df.loc[idx,"Recovered"] = np.nan
                    print(f"Missing {key} for {id} & {year}... setting NaN")
            else:
                df.loc[idx,"Recovered"] = float(data[key].split("\n")[0])

            key = "Estimated Gas Collection Efficiency HH3"
            if data[key].split("\n")[0] == "(":
                if verbose:
                    print(f"Missing {key} for {id} & {year}... setting NaN")
                    df.loc[idx,"Collection_Eff"] = np.nan
            else:
                df.loc[idx,"Collection_Eff"] = float(data[key].split("\n")[0])
            
            key = "Basis for Input Methane Generation Value"
            if data[key] == "Equation HH-1":
                df.loc[idx,"If_Gen_Rec_0"] = False
            elif data[key] == "Equation HH-4":
                df.loc[idx,"If_Gen_Rec_0"] = True
            else:
                if verbose:
                    df.loc[idx,"If_Gen_Rec_0"] = np.nan
                    print(f"Missing {key} for {id} & {year}... setting NaN")

        # Classify the type of report from the landfill
        if np.isnan(df.loc[idx,"Emis_Backward"]):
            df.loc[idx,"If_All_Forward"] = df.loc[idx,"Emis_Reported"]
            df.loc[idx,"If_All_Backward"] = df.loc[idx,"Emis_Reported"]
            df.loc[idx,"If_All_Higher"] = df.loc[idx,"Emis_Reported"]
            if df.loc[idx,"Gas_Collection"] != False:
                # This covers edge cases with collection but w/o both reports
                backwards = df.loc[idx,"Emis_Backward"]
                assert backwards == 0.0 or np.isnan(backwards), f"{id} & {year}"
                df.loc[idx,"Type"] = "Forward"
            else:
                df.loc[idx,"Type"] = "No Gas Collection"
        else:
            df.loc[idx,"If_All_Forward"] = df.loc[idx,"Emis_Forward"]
            df.loc[idx,"If_All_Backward"] = df.loc[idx,"Emis_Backward"]
            df.loc[idx,"If_All_Higher"] = max(df.loc[idx,"Emis_Forward"],
                                              df.loc[idx,"Emis_Backward"])

            reported = df.loc[idx,"Emis_Reported"]
            if abs(df.loc[idx,"Emis_Forward"] - reported) < 25:
                df.loc[idx,"Type"] = "Forward"
            elif abs(df.loc[idx,"Emis_Backward"] - reported) < 25:
                df.loc[idx,"Type"] = "Backward"
            else:
                df.loc[idx,"Type"] = "Other"

    # Use most recent report to get waste-in-place data
    search_year = years[-1]

    # There are specific exceptions to this, specifically for closed landfills
    exceptions = {1002104: 2021, 1002843: 2010, 1011257: 2013, 1003466: 2015,
                  1004255: 2018, 1007007: 2014, 1007906: 2012, 1002135: np.nan,
                  1003592: np.nan, 1005908: np.nan, 1006470: np.nan,
                  1004364: 2020, 1007007: 2015, 1007040: 2021,
                  1002307: 2020, # add 2021 and 2022 WIP data manuallyy
                  1005488: 2020, # add 2021 and 2022 WIP data manually
                  1002511: 2021, # add 2022 WIP data manually
                  1006228: 2021, # add 2022 WIP data manually
                  }
    if id in exceptions.keys():
        search_year = exceptions[id]

    # As long as this is not 1003592, 1005908, 1006470 where this is no data
    if not np.isnan(search_year):

        url = (f"https://ghgdata.epa.gov/ghgp/service/html/"
               f"{search_year}?id={id}&et=undefined")
        soup = BeautifulSoup(recurisve_requests(url).text, "html.parser")

        # Check which year we should start and stop counting waste-in-place
        end_year = str(years[-1])
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) != 0:
                text = cells[0].get_text(strip=True)
                if text == "Starting Year for Accepting Waste":
                    start_year = cells[1].get_text(strip=True)
                if text == "Ending Year for Accepting Waste":
                    end_year = cells[1].get_text(strip=True)

        # Extract waste disposed each year using this regex
        year_quantity_pairs = re.findall(
            r'<b>Reporting Year</b></td><td><b>(\d+)</b></td>\n</tr>\n<tr>\n'
            r'<td>\s*Total Annual Waste Disposal Quantity\s*</td><td>'
            r'\s*(\d*\.?\d+)\s*\(Metric Tons\)', str(soup), re.DOTALL)
        WIP_years, quantities = zip(*year_quantity_pairs)
        WIP_years = np.array([int(y) for y in WIP_years])
        quantities = np.array([float(q) for q in quantities])
        WIP_years_expected = np.array([y for y in range(int(start_year),
                                                        int(end_year)+1)][::-1])

        # Manual fixes from "exceptions" above
        if id == 1002307:
            WIP_years = np.append(WIP_years, [2021, 2022])
            quantities = np.append(quantities, [220947.5672, 202721.7266])
        if id == 1005488:
            WIP_years = np.append(WIP_years, [2021, 2022])
            quantities = np.append(quantities, [205075, 219682])
        if id == 1002511:
            WIP_years = np.append(WIP_years, 2022)
            quantities = np.append(quantities, 304145)
        if id == 1006228:
            WIP_years = np.append(WIP_years, 2022)
            quantities = np.append(quantities, 83300.52)

        if not np.array_equal(WIP_years, WIP_years_expected):
            s1 = sorted(set(WIP_years)-set(WIP_years_expected))
            s2 = sorted(set(WIP_years_expected)-set(WIP_years))
            if verbose:
                print(f"Mismatch for WIP years for id={id}... "
                    f"reported but not expected= {s1}, "
                    f"expected but not reported= {s2}")

        # Get waste-in-place metrics (total and for this year)
        for idx,year in enumerate(years):
            value = np.sum(quantities[WIP_years < year])
            df.loc[idx,"Waste_In_Place_Total"] = value
            if sum(WIP_years == year) == 1:
                assert np.sum(WIP_years == year) == 1
                value = quantities[WIP_years == year][0]
                df.loc[idx,"Waste_Added_This_Yr"] = value
            else:
                df.loc[idx,"Waste_Added_This_Yr"] = 0.0
                if int(year) < int(end_year):
                    if verbose:
                        print(f"For {id}, setting waste added this year to 0.0 "
                            f"despite year = {year} and end year = {end_year}")

    else:
        for idx,year in enumerate(years):
            if verbose:
                print(f"Setting waste-in-place data to NaN for {id} & {year}")
            df.loc[idx,"Waste_In_Place_Total"] = np.nan
            df.loc[idx,"Waste_Added_This_Yr"] = np.nan

    return df