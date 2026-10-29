<?php
/**
 * Plugin Name:       AI Lab Doc Intelligence
 * Description:       Upload a PDF, extract structured fields via the AI Lab worker, review and correct, approve, export CSV.
 * Version:           0.1.0
 * Author:            AI Lab
 * Requires at least: 6.0
 */

if (!defined('ABSPATH')) exit;

register_activation_hook(__FILE__, function () {
    global $wpdb;
    require_once ABSPATH . 'wp-admin/includes/upgrade.php';
    dbDelta("CREATE TABLE {$wpdb->prefix}ai_lab_tasks (
        id BIGINT(20) UNSIGNED NOT NULL AUTO_INCREMENT,
        doc_name VARCHAR(255) NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',
        fields_json LONGTEXT,
        approved_json LONGTEXT,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id)
    ) {$wpdb->get_charset_collate()};");
});

add_action('admin_menu', function () {
    add_menu_page('Doc Intelligence', 'Doc Intelligence', 'manage_options', 'ai-lab-docintel', 'ailab_render_page', 'dashicons-media-document', 26);
    add_submenu_page('ai-lab-docintel', 'Doc Intel Settings', 'Settings', 'manage_options', 'ai-lab-docintel-settings', 'ailab_render_settings');
});

add_action('admin_init', function () {
    register_setting('ailab_settings', 'ailab_worker_url', ['sanitize_callback' => 'esc_url_raw']);
    register_setting('ailab_settings', 'ailab_app_key',    ['sanitize_callback' => 'sanitize_text_field']);
    register_setting('ailab_settings', 'ailab_fields',     ['sanitize_callback' => 'sanitize_textarea_field']);
});

function ailab_fields_parsed() {
    $spec = get_option('ailab_fields', "supplier|the company issuing the document\nquote_date|date of the document\ntotal|total amount");
    $out = [];
    foreach (preg_split('/\r?\n/', $spec) as $line) {
        $line = trim($line);
        if (!$line) continue;
        $parts = array_map('trim', explode('|', $line, 2));
        $out[] = ['name' => $parts[0], 'description' => $parts[1] ?? ''];
    }
    return $out;
}

function ailab_worker_post($path, $body) {
    $url = rtrim(get_option('ailab_worker_url', ''), '/') . $path;
    if (!$url || $url === $path) return ['error' => 'Worker URL not configured (see Settings).'];
    $resp = wp_remote_post($url, [
        'headers' => [
            'Content-Type' => 'application/json',
            'X-Ingest-Key' => get_option('ailab_app_key', ''),
        ],
        'body'    => wp_json_encode($body),
        'timeout' => 120,
    ]);
    if (is_wp_error($resp)) return ['error' => $resp->get_error_message()];
    $code = wp_remote_retrieve_response_code($resp);
    $data = json_decode(wp_remote_retrieve_body($resp), true);
    if ($code >= 400) return ['error' => ($data && $data['error']) ? $data['error'] : "Worker HTTP $code"];
    return $data ?: ['error' => 'Empty worker response'];
}

add_action('wp_ajax_ailab_extract', function () {
    check_ajax_referer('ailab_nonce');
    if (!current_user_can('manage_options')) wp_send_json_error('Forbidden', 403);
    $text   = wp_unslash($_POST['text'] ?? '');
    $result = ailab_worker_post('/extract', [
        'text'   => $text,
        'fields' => ailab_fields_parsed(),
    ]);
    wp_send_json($result);
});

add_action('wp_ajax_ailab_save', function () {
    check_ajax_referer('ailab_nonce');
    if (!current_user_can('manage_options')) wp_send_json_error('Forbidden', 403);
    global $wpdb;
    $approved = wp_unslash($_POST['approved'] ?? '');
    $id = $wpdb->insert("{$wpdb->prefix}ai_lab_tasks", [
        'doc_name'      => sanitize_file_name($_POST['doc_name'] ?? 'document'),
        'status'        => $approved ? 'approved' : 'pending',
        'fields_json'   => wp_unslash($_POST['fields'] ?? ''),
        'approved_json' => $approved ?: null,
    ]);
    wp_send_json(['ok' => (bool) $id, 'id' => $wpdb->insert_id]);
});

add_action('admin_post_ailab_csv', function () {
    if (!current_user_can('manage_options')) wp_die('Forbidden');
    global $wpdb;
    $rows = $wpdb->get_results("SELECT doc_name, approved_json, created_at FROM {$wpdb->prefix}ai_lab_tasks WHERE status = 'approved' ORDER BY id DESC LIMIT 5000", ARRAY_A);
    header('Content-Type: text/csv; charset=utf-8');
    header('Content-Disposition: attachment; filename=ai-lab-extractions.csv');
    $out = fopen('php://output', 'w');
    fputcsv($out, ['doc_name', 'extracted_at']);
    foreach ($rows as $r) {
        $fields = json_decode($r['approved_json'], true) ?: [];
        foreach ($fields as $name => $value) {
            fputcsv($out, [$r['doc_name'], $name, is_scalar($value) ? $value : wp_json_encode($value), $r['created_at']]);
        }
    }
    fclose($out);
    exit;
});

function ailab_render_settings() {
    ?>
    <div class="wrap">
      <h1>Doc Intelligence Settings</h1>
      <form method="post" action="options.php">
        <?php settings_fields('ailab_settings'); ?>
        <table class="form-table">
          <tr><th>Worker URL</th>
              <td><input type="url" name="ailab_worker_url" value="<?php echo esc_attr(get_option('ailab_worker_url', 'https://ai-lab-worker.antwibrand.workers.dev')); ?>" class="regular-text"></td></tr>
          <tr><th>App key</th>
              <td><input type="password" name="ailab_app_key" value="<?php echo esc_attr(get_option('ailab_app_key', '')); ?>" class="regular-text">
                  <p class="description">Stored server-side only; never exposed to the browser.</p></td></tr>
          <tr><th>Fields to extract</th>
              <td><textarea name="ailab_fields" rows="6" class="large-text code"><?php echo esc_textarea(get_option('ailab_fields', '')); ?></textarea>
                  <p class="description">One per line: <code>field_name|description shown to the model</code></p></td></tr>
        </table>
        <?php submit_button(); ?>
      </form>
    </div>
    <?php
}

function ailab_render_page() {
    global $wpdb;
    wp_enqueue_script('pdfjs', 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js', [], null, true);
    wp_enqueue_script('ailab-docintel', plugin_dir_url(__FILE__) . 'docintel.js', ['jquery'], '0.1.0', true);
    wp_localize_script('ailab-docintel', 'AILAB', [
        'ajax'   => admin_url('admin-ajax.php'),
        'nonce'  => wp_create_nonce('ailab_nonce'),
        'fields' => ailab_fields_parsed(),
    ]);
    $tasks = $wpdb->get_results("SELECT * FROM {$wpdb->prefix}ai_lab_tasks ORDER BY id DESC LIMIT 50");
    ?>
    <div class="wrap">
      <h1>Doc Intelligence</h1>
      <div id="ailab-app" style="max-width:860px">
        <div id="ailab-drop" style="border:2px dashed #c3c4c7;border-radius:8px;padding:28px;text-align:center;color:#787c82;cursor:pointer;margin-bottom:10px">
          Drop a PDF here (or click to choose)
        </div>
        <input type="file" id="ailab-file" accept="application/pdf" style="display:none">
        <p><textarea id="ailab-paste" rows="4" class="large-text" placeholder="…or paste raw text directly"></textarea></p>
        <p><button class="button button-primary" id="ailab-extract-btn">Extract fields</button>
           <span id="ailab-status" style="margin-left:10px;color:#787c82"></span></p>
        <div id="ailab-results"></div>
        <hr>
        <h2>Recent tasks</h2>
        <a class="button" href="<?php echo esc_url(admin_url('admin-post.php?action=ailab_csv')); ?>">Export approved (CSV)</a>
        <table class="widefat striped" style="margin-top:10px">
          <thead><tr><th>#</th><th>Document</th><th>Status</th><th>Created</th></tr></thead>
          <tbody>
          <?php foreach ($tasks as $t): ?>
            <tr><td><?php echo esc_html($t->id); ?></td><td><?php echo esc_html($t->doc_name); ?></td>
                <td><?php echo esc_html($t->status); ?></td><td><?php echo esc_html($t->created_at); ?></td></tr>
          <?php endforeach; ?>
          </tbody>
        </table>
      </div>
    </div>
    <?php
}
