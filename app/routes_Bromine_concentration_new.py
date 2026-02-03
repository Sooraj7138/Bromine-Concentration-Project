def register_bromine_concentration_routes(app):
    from flask import render_template, request, redirect, url_for, send_file, flash, jsonify
    import os
    from app.Bromine_concentration_service import run_bromine_concentration
    from Final_concentration_service import forecast_future_C2

    UPLOAD_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], 'evaporation')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # ensures folder exists
    app.config['UPLOAD_FOLDER_EVAP'] = UPLOAD_FOLDER
    ALLOWED_EXTENSIONS = {'csv'}

    # === Helper ===
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/Bromine-Concentration', methods=['GET', 'POST'])
    def BromineConcentration():
        return render_template('Bromine_Concentration.html')

    @app.route('/Bromine-Concentration_BC', methods=['GET', 'POST'])
    def BromineConcentrationbc():
        if request.method == 'POST':
            mode = request.form.get("toggle", "a")  # 'a' = zone, 'b' = pond
            print("🔎 Full form data:", request.form)
            user_start = request.form.get("startMonthEvap")
            user_end = request.form.get("endMonthEvap")
            selected_zone = request.form.get('zone_select') if mode == 'a' else None
            selected_pond = request.form.get('pond_select') if mode == 'b' else None
            in_con_str = request.form.get("initial-concentration", "").strip() if mode == 'a' else None
            if mode == 'a' and (not in_con_str):
                flash("Initial concentration is required.", "error")
                return redirect(request.url)

            if mode == 'a':
                try:
                    in_con = float(in_con_str)
                except ValueError:
                    flash("Invalid initial concentration value.", "error")
                    return redirect(request.url)

            print("SEL ", selected_pond)

            # Upload and save files
            upload_folder = 'uploads'
            os.makedirs(upload_folder, exist_ok=True)
            evap_file = request.files.get('evap_file')
            wl_file = request.files.get('wl_file') if mode == 'a' else None
            c2_file = request.files.get('c2_file') if mode == 'b' else None

            if mode == 'a' and (not evap_file or evap_file.filename == ''):
                flash("Evaporation file is required.", "error")
                return redirect(request.url)

            if mode == 'a' and (not wl_file or wl_file.filename == ''):
                flash("Water-Level file is required for Zone.", "error")
                return redirect(request.url)

            # Save files
            evap_path = os.path.join(upload_folder, evap_file.filename) if evap_file else None
            evap_file.save(evap_path) if evap_file else None
            wl_path = os.path.join(upload_folder, wl_file.filename) if wl_file else None
            c2_path = os.path.join(upload_folder, c2_file.filename) if c2_file else None
            c2_file.save(c2_path) if c2_file else None
            if wl_file:
                wl_file.save(wl_path)
            else:
                None

            if mode == "a":
                df_result, plot_html, plot_v, plot_t = run_bromine_concentration(
                    wl_path, evap_path, None, None, in_con=in_con, mode=mode, pond_name=selected_pond)
            else:
                # df_result, plot_html, plot_v, plot_t = run_bromine_concentration(
                #     wl_path, evap_path, in_path, out_path, in_con=in_con, mode=mode, pond_name=selected_pond)
                df_result, plot_html = forecast_future_C2(user_start, user_end, c2_path, pond_name=selected_pond)

            # Final result display
            style = """
                    <style>
                        table {
                            width: 100%;
                            border-collapse: collapse;
                            text-align: center;
                            color: white;
                        }
                        th, td {
                            text-align: center;
                            vertical-align: middle;
                        }
                    </style>
                """
            # table_html = style + df_result.to_html(classes="table table-striped table-bordered", index=False)

            # Convert DataFrame to JSON for AG Grid
            table_data = df_result.to_dict(orient='records')  # list of dicts
            columns = [{"headerName": col, "field": col} for col in df_result.columns]

            # Build the label
            prediction_label = ""
            if mode == 'a':
                prediction_label = "Calculation for Zone-B"
            elif mode == 'b' and selected_pond:
                prediction_label = f"{'Calculation'} for {selected_pond}"

            if mode == 'a':
                return render_template(
                    "Bromine_Concentration.html",
                    # table_html=table_html,
                    table_data=table_data,
                    columns=columns,
                    plot_html=plot_html,
                    plot_v=plot_v,
                    plot_t=plot_t,
                    prediction_label=prediction_label,
                    show_output=True
                )
            else:
                return render_template(
                    "Bromine_Concentration.html",
                    table_data=table_data,
                    columns=columns,
                    plot_html=plot_html,
                    prediction_label=prediction_label,
                    show_output=True
                )

        return render_template("Bromine_Concentration.html")


    # @app.route('/Bromine-Concentration_BC', methods=['GET', 'POST'])
    # def BromineConcentrationbc():
    #     if request.method == 'POST':
    #         mode = request.form.get("toggle", "a")  # 'a' = zone, 'b' = pond
    #         print("🔎 Full form data:", request.form)
    #         # start_month = request.form.get("start_month")  # "2024-03"
    #         # end_month = request.form.get("end_month")
    #         # category = request.form.get("toggleCat", "pred")  # 'pred' or 'calc'
    #         selected_zone = request.form.get('zone_select') if mode == 'a' else None
    #         selected_pond = request.form.get('pond_select') if mode == 'b' else None
    #         in_con_str = request.form.get("initial-concentration", "").strip()
    #         if not in_con_str:
    #             flash("Initial concentration is required.", "error")
    #             return redirect(request.url)
    #         try:
    #             in_con = float(in_con_str)
    #         except ValueError:
    #             flash("Invalid initial concentration value.", "error")
    #             return redirect(request.url)
    #
    #         print("SEL ",selected_pond)
    #
    #         # Upload and save files
    #         upload_folder = 'uploads'
    #         os.makedirs(upload_folder, exist_ok=True)
    #         evap_file = request.files.get('evap_file')
    #         wl_file = request.files.get('wl_file') if mode == 'a' else None
    #         in_file = request.files.get('in_file') if mode == 'b' else None
    #         out_file = request.files.get('out_file') if mode == 'b' else None
    #
    #         if not evap_file or evap_file.filename == '':
    #             flash("Evaporation file is required.", "error")
    #             return redirect(request.url)
    #
    #         if mode == 'a' and (not wl_file or wl_file.filename == ''):
    #             flash("Water-Level file is required for Zone.", "error")
    #             return redirect(request.url)
    #
    #         # Save files
    #         evap_path = os.path.join(upload_folder, evap_file.filename)
    #         evap_file.save(evap_path)
    #         wl_path = os.path.join(upload_folder, wl_file.filename) if wl_file else None
    #         if wl_file:
    #             wl_file.save(wl_path)
    #         else:
    #             None
    #         if mode == "b":
    #             if not in_file or in_file.filename == '':
    #                 flash("Brine In file is required for Pond.", "error")
    #                 return redirect(request.url)
    #             in_path = os.path.join(upload_folder, in_file.filename)
    #             in_file.save(in_path)
    #
    #             if not out_file or out_file.filename == '':
    #                 flash("Brine Out file is required for Pond.", "error")
    #                 return redirect(request.url)
    #             out_path = os.path.join(upload_folder, out_file.filename)
    #             out_file.save(out_path)
    #
    #             # ⬇️ CONDITIONAL LOGIC based on category
    #         # if category == 'pred':
    #         #     # Get dates only in prediction mode
    #         #     df_result, plot_html, plot_v, plot_t = run_bromine_concentration(
    #         #         wl_path, evap_path, in_con=in_con, mode=mode, pond_name=selected_pond, forecast=True,
    #         #         user_start=start_month, user_end=end_month)
    #         #
    #         # else:  # category == 'calc'
    #         #     df_result, plot_html, plot_v, plot_t  = run_bromine_concentration(
    #         #         wl_path, evap_path, in_con=in_con, mode=mode, pond_name=selected_pond, forecast=False,
    #         #         user_start=start_month, user_end=end_month)
    #
    #         if mode == "a":
    #             df_result, plot_html, plot_v, plot_t = run_bromine_concentration(
    #                 wl_path, evap_path, None, None, in_con=in_con, mode=mode, pond_name=selected_pond)
    #         else:
    #             df_result, plot_html, plot_v, plot_t = run_bromine_concentration(
    #                 wl_path, evap_path, in_path, out_path, in_con=in_con, mode=mode, pond_name=selected_pond)
    #
    #         # Final result display
    #         style = """
    #             <style>
    #                 table {
    #                     width: 100%;
    #                     border-collapse: collapse;
    #                     text-align: center;
    #                     color: white;
    #                 }
    #                 th, td {
    #                     text-align: center;
    #                     vertical-align: middle;
    #                 }
    #             </style>
    #         """
    #         # table_html = style + df_result.to_html(classes="table table-striped table-bordered", index=False)
    #
    #         # Convert DataFrame to JSON for AG Grid
    #         table_data = df_result.to_dict(orient='records')  # list of dicts
    #         columns = [{"headerName": col, "field": col} for col in df_result.columns]
    #
    #         # Build the label
    #         prediction_label = ""
    #         if mode == 'a':
    #             prediction_label = "Calculation for Zone-B"
    #         elif mode == 'b' and selected_pond:
    #             prediction_label = f"{'Calculation'} for {selected_pond}"
    #
    #         return render_template(
    #             "Bromine_Concentration.html",
    #             # table_html=table_html,
    #             table_data=table_data,
    #             columns=columns,
    #             plot_html=plot_html,
    #             plot_v=plot_v,
    #             plot_t=plot_t,
    #             prediction_label=prediction_label,
    #             show_output=True
    #         )
    #
    #     return render_template("Bromine_Concentration.html")

    @app.route("/download_predictions_c2")
    def download_predictions_C2():
        try:
            if not os.path.exists(
                    "D:/Sooraj\Project_Bromine_Concentration/app/outputs/bromine_concentration_output.csv"):
                flash("❌ No predictions available to download.","error")
                return redirect(url_for("BromineConcentrationbc"))

            return send_file(
                "D:/Sooraj\Project_Bromine_Concentration/app/outputs/bromine_concentration_output.csv",
                mimetype="text/csv",
                as_attachment=True,
                download_name="Bromine_Prediction.csv"
            )
        except Exception as e:
            print("Download error:", e)
            flash("❌ Failed to download predictions.","error")
            return redirect(url_for("BromineConcentrationbc"))